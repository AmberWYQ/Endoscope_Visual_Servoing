"""
Testing Mode: Evaluate trained models without hardware
Runs in simulation with visualization
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import json
import cv2
from datetime import datetime
from typing import Dict, Optional
import argparse

# Fix Qt issue for headless servers
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, Config
from models.network import EndoscopeTrackingNetwork
from simulator.endoscope_sim import make_env


class TestRunner:
    """
    Run tests on trained models in simulation
    """
    
    def __init__(self, 
                 config: Config,
                 checkpoint_path: str,
                 output_dir: str = "./test_results"):
        self.config = config
        self.device = torch.device(config.training.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = EndoscopeTrackingNetwork(config).to(self.device)
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'actor_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['actor_state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Create environment
        self.env = make_env(config)
        
        print("Model loaded successfully!")
    
    def select_action(self, obs: Dict, deterministic: bool = True) -> np.ndarray:
        """Get action from model"""
        with torch.no_grad():
            image = torch.tensor(
                obs['image'].transpose(2, 0, 1)[None] / 255.0,
                dtype=torch.float32, device=self.device
            )
            trajectory = torch.tensor(
                obs['trajectory'][None],
                dtype=torch.float32, device=self.device
            )
            em_state = torch.tensor(
                obs['em_state'][None],
                dtype=torch.float32, device=self.device
            )
            motor_state = torch.tensor(
                obs['motor_state'][None],
                dtype=torch.float32, device=self.device
            )
            image_error = torch.tensor(
                obs['image_error'][None],
                dtype=torch.float32, device=self.device
            )
            
            output = self.model(
                image, trajectory, em_state, motor_state, image_error,
                deterministic=deterministic
            )
            
            action = output['action'].cpu().numpy()[0]
            jacobian = output['jacobian'].cpu().numpy()[0]
            
        return action, jacobian
    
    def run_episode(self, 
                    episode_id: int = 0,
                    max_steps: int = 500,
                    save_video: bool = True,
                    visualize: bool = False) -> Dict:
        """Run single test episode"""
        obs, info = self.env.reset()
        
        # Storage
        frames = []
        rewards = []
        errors = []
        actions = []
        jacobians = []
        
        print(f"Running episode {episode_id}...")
        
        for step in range(max_steps):
            # Get action
            action, jacobian = self.select_action(obs)
            
            # Step
            next_obs, reward, term, trunc, info = self.env.step(action)
            
            # Store data
            frames.append(obs['image'].copy())
            rewards.append(reward)
            errors.append(info['tracking_error'])
            actions.append(action.copy())
            jacobians.append(jacobian.copy())
            
            # Visualize
            if visualize:
                vis_frame = self._create_visualization(obs, info, action, jacobian, step)
                cv2.imshow('Test', vis_frame)
                key = cv2.waitKey(50)
                if key == ord('q'):
                    break
            
            if term or trunc:
                break
            
            obs = next_obs
        
        if visualize:
            cv2.destroyAllWindows()
        
        # Save video
        if save_video and frames:
            self._save_video(frames, episode_id, errors)
        
        # Compute metrics
        results = {
            'episode_id': episode_id,
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards),
            'mean_error': np.mean(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'final_error': errors[-1] if errors else 0,
            'num_steps': len(rewards),
            'success': np.mean(errors) < 50  # Success if avg error < 50 pixels
        }
        
        print(f"Episode {episode_id} results:")
        print(f"  Total reward: {results['total_reward']:.2f}")
        print(f"  Mean error: {results['mean_error']:.2f} pixels")
        print(f"  Success: {results['success']}")
        
        return results
    
    def _create_visualization(self, obs: Dict, info: Dict, action: np.ndarray,
                             jacobian: np.ndarray, step: int) -> np.ndarray:
        """Create visualization frame with overlays"""
        frame = obs['image'].copy()
        
        # Draw target
        target_uv = info['target_uv'].astype(int)
        cv2.circle(frame, tuple(target_uv), 10, (0, 0, 255), 2)
        
        # Draw center
        cx, cy = self.config.camera.width // 2, self.config.camera.height // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Draw error vector
        cv2.arrowedLine(frame, (cx, cy), tuple(target_uv), (255, 255, 0), 2)
        
        # Draw info panel
        info_panel = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)
        
        texts = [
            f"Step: {step}",
            f"Error: {info['tracking_error']:.1f}px",
            f"Action: [{action[0]:.1f}, {action[1]:.1f}]",
            f"J: [[{jacobian[0,0]:.2f}, {jacobian[0,1]:.2f}], [{jacobian[1,0]:.2f}, {jacobian[1,1]:.2f}]]"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(info_panel, text, (10, 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine
        combined = np.vstack([frame, info_panel])
        
        return combined
    
    def _save_video(self, frames: list, episode_id: int, errors: list):
        """Save episode as video"""
        output_path = self.output_dir / f"episode_{episode_id}.mp4"
        
        # Create video with info overlay
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, 20, (w, h + 100))
        
        for i, frame in enumerate(frames):
            # Add info panel
            info_panel = np.zeros((100, w, 3), dtype=np.uint8)
            error = errors[i] if i < len(errors) else 0
            
            cv2.putText(info_panel, f"Frame: {i}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Error: {error:.1f}px", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            combined = np.vstack([frame, info_panel])
            writer.write(combined)
        
        writer.release()
        print(f"Saved video: {output_path}")
    
    def run_benchmark(self, 
                      num_episodes: int = 20,
                      save_videos: bool = False) -> Dict:
        """Run full benchmark"""
        print(f"\nRunning benchmark with {num_episodes} episodes...")
        
        all_results = []
        
        for ep in range(num_episodes):
            results = self.run_episode(
                episode_id=ep,
                save_video=save_videos,
                visualize=False
            )
            all_results.append(results)
        
        # Aggregate results
        benchmark = {
            'num_episodes': num_episodes,
            'timestamp': datetime.now().isoformat(),
            'mean_reward': np.mean([r['total_reward'] for r in all_results]),
            'std_reward': np.std([r['total_reward'] for r in all_results]),
            'mean_error': np.mean([r['mean_error'] for r in all_results]),
            'std_error': np.std([r['mean_error'] for r in all_results]),
            'success_rate': np.mean([r['success'] for r in all_results]),
            'episodes': all_results
        }
        
        # Save results
        results_path = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
        
        print(f"\n{'='*50}")
        print("BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Episodes: {num_episodes}")
        print(f"Mean Reward: {benchmark['mean_reward']:.2f} ± {benchmark['std_reward']:.2f}")
        print(f"Mean Error: {benchmark['mean_error']:.2f} ± {benchmark['std_error']:.2f} pixels")
        print(f"Success Rate: {benchmark['success_rate']*100:.1f}%")
        print(f"\nResults saved to: {results_path}")
        
        return benchmark
    
    def compare_controllers(self, num_episodes: int = 10) -> Dict:
        """Compare learned policy vs baseline controllers"""
        results = {
            'learned_policy': [],
            'proportional': [],
            'jacobian_based': []
        }
        
        print("\nComparing controllers...")
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            
            # Test learned policy
            policy_rewards = []
            policy_errors = []
            obs_copy, _ = self.env.reset()
            
            for step in range(200):
                action, _ = self.select_action(obs_copy)
                obs_copy, reward, term, trunc, info = self.env.step(action)
                policy_rewards.append(reward)
                policy_errors.append(info['tracking_error'])
                if term or trunc:
                    break
            
            results['learned_policy'].append({
                'reward': sum(policy_rewards),
                'mean_error': np.mean(policy_errors)
            })
            
            # Test proportional controller
            obs, info = self.env.reset()
            prop_rewards = []
            prop_errors = []
            
            for step in range(200):
                error = obs['image_error']
                action = -0.1 * error  # Simple P controller
                action = np.clip(action, -50, 50)
                obs, reward, term, trunc, info = self.env.step(action)
                prop_rewards.append(reward)
                prop_errors.append(info['tracking_error'])
                if term or trunc:
                    break
            
            results['proportional'].append({
                'reward': sum(prop_rewards),
                'mean_error': np.mean(prop_errors)
            })
            
            # Test Jacobian-based controller
            obs, info = self.env.reset()
            jac_rewards = []
            jac_errors = []
            
            for step in range(200):
                error = obs['image_error']
                jacobian = info['jacobian_gt']
                try:
                    J_inv = np.linalg.pinv(jacobian)
                    action = -0.3 * J_inv @ error
                    action = np.clip(action, -50, 50)
                except:
                    action = np.zeros(2)
                obs, reward, term, trunc, info = self.env.step(action)
                jac_rewards.append(reward)
                jac_errors.append(info['tracking_error'])
                if term or trunc:
                    break
            
            results['jacobian_based'].append({
                'reward': sum(jac_rewards),
                'mean_error': np.mean(jac_errors)
            })
        
        # Print comparison
        print(f"\n{'='*60}")
        print("CONTROLLER COMPARISON")
        print(f"{'='*60}")
        
        for name, data in results.items():
            rewards = [d['reward'] for d in data]
            errors = [d['mean_error'] for d in data]
            print(f"\n{name}:")
            print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"  Error:  {np.mean(errors):.2f} ± {np.std(errors):.2f} pixels")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test Mode - No Hardware")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--visualize', action='store_true', help='Show visualization (requires display)')
    parser.add_argument('--save-videos', action='store_true', help='Save episode videos')
    parser.add_argument('--benchmark', action='store_true', help='Run full benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare with baseline controllers')
    parser.add_argument('--output-dir', type=str, default='./test_results', help='Output directory')
    args = parser.parse_args()
    
    config = get_config()
    
    runner = TestRunner(config, args.checkpoint, args.output_dir)
    
    if args.compare:
        runner.compare_controllers(args.num_episodes)
    elif args.benchmark:
        runner.run_benchmark(args.num_episodes, args.save_videos)
    else:
        # Run single episode with visualization
        runner.run_episode(
            episode_id=0,
            save_video=args.save_videos,
            visualize=args.visualize
        )


if __name__ == "__main__":
    main()

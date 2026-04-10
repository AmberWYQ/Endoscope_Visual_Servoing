"""
Safety Manager for Visual Servoing Pipeline

Handles:
- Action filtering and smoothing
- No-detection fallback behavior
- Low-confidence gain reduction
- Velocity limiting and rate limiting
- Emergency stop conditions
- Search behavior when target is lost

All safety checks are applied between the control output and motor commands.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class SafetyState(Enum):
    """Current safety state"""
    NORMAL = "normal"              # Normal tracking operation
    LOW_CONFIDENCE = "low_conf"    # Tracking but with reduced gain
    HOLDING = "holding"            # Holding position (no detection)
    SEARCHING = "searching"        # Executing search pattern
    STOPPED = "stopped"            # Emergency stop
    ERROR = "error"                # Error state


@dataclass
class SafetyOutput:
    """Output from safety manager"""
    # Final motor command [m1, m2] in normalized [-1, 1]
    action: np.ndarray
    
    # Current safety state
    state: SafetyState
    
    # Whether output is safe to execute
    is_safe: bool
    
    # Gain applied (for diagnostics)
    applied_gain: float
    
    # Reason for current state
    reason: str
    
    # Search direction if in search mode
    search_direction: Optional[np.ndarray] = None


class SafetyManager:
    """
    Manages safety constraints and fallback behaviors for visual servoing.
    
    Usage:
        safety = SafetyManager(config.safety)
        
        # In control loop:
        safe_output = safety.process(
            raw_action=control_output.action,
            detection_result=detection,
            confidence=detection.confidence,
        )
        
        if safe_output.is_safe:
            send_to_motors(safe_output.action)
    """
    
    def __init__(self, config):
        """
        Initialize safety manager.
        
        Args:
            config: SafetyConfig or full IntegratedConfig
        """
        # Extract safety config
        if hasattr(config, 'safety'):
            self.config = config.safety
        else:
            self.config = config
        
        # State
        self.state = SafetyState.NORMAL
        self.last_action = np.zeros(2)
        self.filtered_action = np.zeros(2)
        
        # Counters
        self.no_detection_count = 0
        self.low_confidence_count = 0
        self.error_count = 0
        
        # Search pattern state
        self.search_start_time = 0.0
        self.search_phase = 0.0
        
        # Timing
        self.last_update_time = time.time()
    
    def process(
        self,
        raw_action: np.ndarray,
        detection_valid: bool,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> SafetyOutput:
        """
        Process raw action through safety checks.
        
        Args:
            raw_action: Raw action from control model [m1, m2] in [-1, 1]
            detection_valid: Whether detection is valid (not no_detection)
            confidence: Detection confidence (0-1)
            timestamp: Current timestamp (or uses time.time())
        
        Returns:
            SafetyOutput with filtered action and state
        """
        if timestamp is None:
            timestamp = time.time()
        
        dt = timestamp - self.last_update_time
        self.last_update_time = timestamp
        
        # Start with raw action
        action = raw_action.copy()
        applied_gain = 1.0
        reason = ""
        
        # Check for no detection
        if not detection_valid:
            self.no_detection_count += 1
            self.low_confidence_count = 0
            
            if self.config.no_detection_action == 'stop':
                self.state = SafetyState.STOPPED
                action = np.zeros(2)
                reason = "No detection - stopped"
            
            elif self.config.no_detection_action == 'hold':
                self.state = SafetyState.HOLDING
                action = np.zeros(2)
                reason = f"No detection - holding (count={self.no_detection_count})"
            
            elif self.config.no_detection_action == 'search':
                if self.config.search_pattern_enabled:
                    self.state = SafetyState.SEARCHING
                    action = self._get_search_action(timestamp)
                    reason = f"No detection - searching"
                else:
                    self.state = SafetyState.HOLDING
                    action = np.zeros(2)
                    reason = "No detection - holding (search disabled)"
            
        else:
            # Valid detection - check confidence
            self.no_detection_count = 0
            
            if confidence < self.config.low_confidence_threshold:
                self.low_confidence_count += 1
                self.state = SafetyState.LOW_CONFIDENCE
                applied_gain = self.config.low_confidence_gain_reduction
                action = action * applied_gain
                reason = f"Low confidence ({confidence:.2f}) - reduced gain"
            else:
                self.low_confidence_count = 0
                self.state = SafetyState.NORMAL
                reason = "Normal tracking"
        
        # Apply deadzone
        action = self._apply_deadzone(action)
        
        # Apply velocity limiting
        action = self._apply_velocity_limit(action, dt)
        
        # Apply smoothing
        if self.config.enable_smoothing:
            action = self._apply_smoothing(action)
        
        # Final clip to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Check for emergency conditions
        if self.error_count >= self.config.max_consecutive_errors:
            self.state = SafetyState.ERROR
            action = np.zeros(2)
            reason = f"Emergency stop - too many errors ({self.error_count})"
        
        # Update state
        self.last_action = action.copy()
        
        # Determine if safe to execute
        is_safe = self.state not in [SafetyState.ERROR, SafetyState.STOPPED]
        
        return SafetyOutput(
            action=action,
            state=self.state,
            is_safe=is_safe,
            applied_gain=applied_gain,
            reason=reason,
            search_direction=action if self.state == SafetyState.SEARCHING else None,
        )
    
    def _apply_deadzone(self, action: np.ndarray) -> np.ndarray:
        """Apply deadzone to action"""
        # This is typically applied to error, but we can also apply to action
        # For now, just return action as-is since deadzone is handled in control
        return action
    
    def _apply_velocity_limit(self, action: np.ndarray, dt: float) -> np.ndarray:
        """Limit rate of change of action"""
        if dt <= 0:
            return action
        
        # Compute desired change
        delta = action - self.last_action
        
        # Limit change per timestep
        max_change = self.config.max_velocity_change * dt
        
        # Clip change
        delta_clipped = np.clip(delta, -max_change, max_change)
        
        return self.last_action + delta_clipped
    
    def _apply_smoothing(self, action: np.ndarray) -> np.ndarray:
        """Apply low-pass filter smoothing"""
        alpha = self.config.smoothing_alpha
        self.filtered_action = alpha * self.filtered_action + (1 - alpha) * action
        return self.filtered_action.copy()
    
    def _get_search_action(self, timestamp: float) -> np.ndarray:
        """Generate search pattern action"""
        if self.no_detection_count == 1:
            # Just started searching
            self.search_start_time = timestamp
        
        # Time since search started
        t = timestamp - self.search_start_time
        
        # Simple sinusoidal search pattern
        period = self.config.search_period
        velocity = self.config.search_velocity / 100.0  # Normalize to [-1, 1]
        
        # Phase shifts for different motors create spiral-like pattern
        m1 = velocity * np.sin(2 * np.pi * t / period)
        m2 = velocity * np.cos(2 * np.pi * t / period)
        
        return np.array([m1, m2])
    
    def report_error(self):
        """Report an error (e.g., motor communication failure)"""
        self.error_count += 1
    
    def clear_error(self):
        """Clear error state"""
        self.error_count = 0
        if self.state == SafetyState.ERROR:
            self.state = SafetyState.NORMAL
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        self.state = SafetyState.STOPPED
        self.filtered_action = np.zeros(2)
        self.last_action = np.zeros(2)
    
    def resume(self):
        """Resume from stop or error state"""
        if self.state in [SafetyState.STOPPED, SafetyState.ERROR]:
            self.state = SafetyState.NORMAL
            self.error_count = 0
    
    def reset(self):
        """Reset all state"""
        self.state = SafetyState.NORMAL
        self.last_action = np.zeros(2)
        self.filtered_action = np.zeros(2)
        self.no_detection_count = 0
        self.low_confidence_count = 0
        self.error_count = 0
        self.search_start_time = 0.0
        self.last_update_time = time.time()
    
    def get_state_info(self) -> dict:
        """Get current state information"""
        return {
            'state': self.state.value,
            'no_detection_count': self.no_detection_count,
            'low_confidence_count': self.low_confidence_count,
            'error_count': self.error_count,
            'last_action': self.last_action.tolist(),
        }


class ActionFilter:
    """
    Additional filtering utilities for control actions.
    
    Can be used standalone or in addition to SafetyManager.
    """
    
    @staticmethod
    def apply_deadzone(error: np.ndarray, deadzone: float = 10.0) -> np.ndarray:
        """
        Apply deadzone to pixel error.
        
        Args:
            error: Pixel error [ex, ey]
            deadzone: Deadzone in pixels
        
        Returns:
            Filtered error with deadzone applied
        """
        result = error.copy()
        result[np.abs(result) < deadzone] = 0
        return result
    
    @staticmethod
    def normalize_error(
        error: np.ndarray,
        img_width: int = 640,
        img_height: int = 480,
    ) -> np.ndarray:
        """
        Normalize pixel error to [-1, 1] range.
        
        Args:
            error: Pixel error [ex, ey]
            img_width: Image width
            img_height: Image height
        
        Returns:
            Normalized error
        """
        norm_x = img_width / 2
        norm_y = img_height / 2
        return np.array([error[0] / norm_x, error[1] / norm_y])
    
    @staticmethod
    def exponential_smoothing(
        current: np.ndarray,
        previous: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Apply exponential smoothing.
        
        Args:
            current: Current value
            previous: Previous filtered value
            alpha: Smoothing factor (0 = no smoothing, 1 = no update)
        
        Returns:
            Smoothed value
        """
        return alpha * previous + (1 - alpha) * current
    
    @staticmethod
    def rate_limit(
        current: np.ndarray,
        previous: np.ndarray,
        max_rate: float,
        dt: float = 0.05,
    ) -> np.ndarray:
        """
        Limit rate of change.
        
        Args:
            current: Current value
            previous: Previous value
            max_rate: Maximum rate of change per second
            dt: Time step
        
        Returns:
            Rate-limited value
        """
        delta = current - previous
        max_delta = max_rate * dt
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        return previous + delta_clipped


if __name__ == '__main__':
    # Test safety manager
    print("Testing SafetyManager...")
    
    from dataclasses import dataclass
    
    @dataclass
    class TestSafetyConfig:
        no_detection_action: str = 'hold'
        low_confidence_threshold: float = 0.5
        low_confidence_gain_reduction: float = 0.5
        enable_smoothing: bool = True
        smoothing_alpha: float = 0.3
        max_velocity_change: float = 20.0
        search_pattern_enabled: bool = True
        search_velocity: float = 10.0
        search_period: float = 2.0
        max_consecutive_errors: int = 50
    
    config = TestSafetyConfig()
    safety = SafetyManager(config)
    
    # Test normal operation
    print("\n1. Normal operation:")
    action = np.array([0.5, -0.3])
    result = safety.process(action, detection_valid=True, confidence=0.8)
    print(f"   State: {result.state.value}, Action: {result.action}, Reason: {result.reason}")
    
    # Test low confidence
    print("\n2. Low confidence:")
    result = safety.process(action, detection_valid=True, confidence=0.3)
    print(f"   State: {result.state.value}, Action: {result.action}, Reason: {result.reason}")
    
    # Test no detection (hold mode)
    print("\n3. No detection (hold):")
    for i in range(3):
        result = safety.process(action, detection_valid=False, confidence=0.0)
        print(f"   {i}: State: {result.state.value}, Action: {result.action}")
    
    # Test search mode
    safety.reset()
    safety.config.no_detection_action = 'search'
    print("\n4. No detection (search):")
    t = 0.0
    for i in range(5):
        result = safety.process(action, detection_valid=False, confidence=0.0, timestamp=t)
        print(f"   t={t:.1f}: State: {result.state.value}, Action: {result.action}")
        t += 0.1
    
    # Test emergency stop
    safety.reset()
    print("\n5. Emergency stop:")
    for i in range(55):
        safety.report_error()
    result = safety.process(action, detection_valid=True, confidence=0.8)
    print(f"   State: {result.state.value}, Action: {result.action}, Safe: {result.is_safe}")
    
    print("\nSafety manager test complete!")

"""
UI Display Module for Visual Servoing Control

Provides visualization functions for:
- Bounding box overlay
- Pixel error arrow (from image center to bbox center)
- Command direction indicator
- Control status display
"""

import pygame
import numpy as np
import cv2


def draw_arrow(surface, start, end, color, thickness=2, tip_length=0.3):
    """
    Draw an arrow from start to end point.
    
    Args:
        surface: Pygame surface to draw on
        start: (x, y) start point
        end: (x, y) end point
        color: RGB color tuple
        thickness: Line thickness
        tip_length: Proportion of arrow length for tip
    """
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    
    # Draw the main line
    pygame.draw.line(surface, color, start, end, thickness)
    
    # Calculate arrow tip
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx*dx + dy*dy)
    
    if length < 5:  # Too short to draw tip
        return
    
    # Unit vector
    ux, uy = dx / length, dy / length
    
    # Perpendicular vector
    px, py = -uy, ux
    
    # Tip size
    tip_size = length * tip_length
    
    # Arrow tip points
    tip1 = (int(end[0] - tip_size * ux + tip_size * 0.5 * px),
            int(end[1] - tip_size * uy + tip_size * 0.5 * py))
    tip2 = (int(end[0] - tip_size * ux - tip_size * 0.5 * px),
            int(end[1] - tip_size * uy - tip_size * 0.5 * py))
    
    # Draw arrow tip
    pygame.draw.polygon(surface, color, [end, tip1, tip2])


def draw_direction_indicator(surface, x, y, command, scale=30):
    """
    Draw a direction indicator showing the bending command.
    
    Args:
        surface: Pygame surface
        x, y: Center position
        command: (d1, d2) normalized command in [-1, 1]
        scale: Size scale
    """
    d1, d2 = command
    
    # Background circle
    pygame.draw.circle(surface, (50, 50, 50), (x, y), scale + 5)
    pygame.draw.circle(surface, (100, 100, 100), (x, y), scale + 5, 2)
    
    # Draw axis lines
    pygame.draw.line(surface, (80, 80, 80), (x - scale, y), (x + scale, y), 1)
    pygame.draw.line(surface, (80, 80, 80), (x, y - scale), (x, y + scale), 1)
    
    # Draw command vector
    end_x = x + int(d1 * scale)
    end_y = y - int(d2 * scale)  # Flip y for display (positive up)
    
    if abs(d1) > 0.01 or abs(d2) > 0.01:
        draw_arrow(surface, (x, y), (end_x, end_y), (0, 255, 0), 3)
    
    # Draw center dot
    pygame.draw.circle(surface, (255, 255, 255), (x, y), 3)


def draw_joystick_arrows(surface, x, y, vectors):
    """
    Draw the 4-direction arrow indicator (from original ui_display.py).
    
    Args:
        surface: Pygame surface
        x, y: Center position
        vectors: (vec_ud, vec_lr) up/down and left/right values [-1, 1]
    """
    vec_ud, vec_lr = vectors
    
    color_active = (0, 0, 255)      # Blue when active
    color_inactive = (50, 50, 50)   # Dark gray when inactive
    size = 20
    offset = 30
    
    # Center square
    pygame.draw.rect(surface, (0, 0, 150), (x - 10, y - 10, 20, 20))

    def draw_poly(pts, alpha):
        col = color_active if alpha > 0.1 else color_inactive
        width = 0 if alpha > 0.1 else 2
        pygame.draw.polygon(surface, col, pts, width)

    # Up arrow
    draw_poly([(x, y-offset-size), (x-size//2, y-offset), (x+size//2, y-offset)], 
              max(0, vec_ud))
    # Down arrow
    draw_poly([(x, y+offset+size), (x-size//2, y+offset), (x+size//2, y+offset)], 
              abs(vec_ud) if vec_ud < 0 else 0)
    # Left arrow
    draw_poly([(x-offset-size, y), (x-offset, y-size//2), (x-offset, y+size//2)], 
              abs(vec_lr) if vec_lr < 0 else 0)
    # Right arrow
    draw_poly([(x+offset+size, y), (x+offset, y-size//2), (x+offset, y+size//2)], 
              max(0, vec_lr))


def draw_bbox(surface, bbox, color, thickness=2):
    """
    Draw a bounding box on the surface.
    
    Args:
        surface: Pygame surface
        bbox: [x, y, w, h] format
        color: RGB color tuple
        thickness: Line thickness
    """
    if bbox is None:
        return
    
    x, y, w, h = bbox
    rect = pygame.Rect(int(x), int(y), int(w), int(h))
    pygame.draw.rect(surface, color, rect, thickness)
    
    # Draw center point
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    pygame.draw.circle(surface, color, (cx, cy), 4)


def draw_center_cross(surface, x, y, size=20, color=(0, 255, 0), thickness=2):
    """
    Draw a crosshair at the specified position.
    
    Args:
        surface: Pygame surface
        x, y: Center position
        size: Arm length
        color: RGB color
        thickness: Line thickness
    """
    pygame.draw.line(surface, color, (x - size, y), (x + size, y), thickness)
    pygame.draw.line(surface, color, (x, y - size), (x, y + size), thickness)


def draw_error_arrow(surface, center, bbox_center, color=(255, 0, 255), scale=1.0):
    """
    Draw arrow from bbox center to image center (showing where target needs to go).
    
    Args:
        surface: Pygame surface
        center: (x, y) image center
        bbox_center: (x, y) bbox center
        color: RGB color
        scale: Arrow length scale factor
    """
    if bbox_center is None:
        return
    
    cx, cy = center
    bx, by = bbox_center
    
    # Calculate error vector (pointing from bbox to center)
    dx = cx - bx
    dy = cy - by
    
    # Scale the arrow
    end_x = bx + dx * scale
    end_y = by + dy * scale
    
    draw_arrow(surface, bbox_center, (end_x, end_y), color, 2, 0.2)


def numpy_to_pygame(frame_rgb, target_size=None):
    """
    Convert numpy RGB array to pygame surface.
    
    Args:
        frame_rgb: Numpy array (H, W, 3) in RGB format
        target_size: Optional (width, height) to resize to
    
    Returns:
        Pygame surface
    """
    if frame_rgb is None:
        return None
    
    # Create surface from array
    surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    
    if target_size is not None:
        surf = pygame.transform.scale(surf, target_size)
    
    return surf


def draw_status_panel(surface, fonts, state_dict, start_pos=(20, 20)):
    """
    Draw status information panel.
    
    Args:
        surface: Pygame surface
        fonts: Tuple of (large_font, mid_font, small_font)
        state_dict: Dictionary with status information
        start_pos: (x, y) top-left position
    """
    f_large, f_mid, f_small = fonts
    x, y = start_pos
    line_height = 25
    
    # Mode indicator
    mode = state_dict.get('mode', 'SIMULATION')
    mode_color = (0, 255, 0) if mode == 'REAL_ROBOT' else (255, 200, 0)
    surface.blit(f_mid.render(f"Mode: {mode}", True, mode_color), (x, y))
    y += line_height
    
    # Tracking status
    status = state_dict.get('tracking_status', 'WAITING')
    if status == 'TRACKING':
        status_color = (0, 255, 0)
    elif status == 'LOST':
        status_color = (255, 255, 0)
    else:
        status_color = (150, 150, 150)
    surface.blit(f_mid.render(f"Status: {status}", True, status_color), (x, y))
    y += line_height
    
    # Confidence
    conf = state_dict.get('confidence', 0.0)
    surface.blit(f_small.render(f"Confidence: {conf:.2f}", True, (200, 200, 200)), (x, y))
    y += line_height - 5
    
    # FPS
    fps = state_dict.get('fps', 0.0)
    surface.blit(f_small.render(f"FPS: {fps:.1f}", True, (200, 200, 200)), (x, y))
    y += line_height
    
    # Pixel error
    error = state_dict.get('pixel_error', (0, 0))
    surface.blit(f_small.render(f"Error: ({error[0]:.0f}, {error[1]:.0f}) px", True, (200, 200, 200)), (x, y))
    y += line_height - 5
    
    # Command
    cmd = state_dict.get('command', (0, 0))
    surface.blit(f_small.render(f"Cmd: ({cmd[0]:.2f}, {cmd[1]:.2f})", True, (200, 200, 200)), (x, y))
    y += line_height - 5
    
    # Motor speeds
    motor = state_dict.get('motor_speeds', (0, 0))
    surface.blit(f_small.render(f"Motor: ({motor[0]}, {motor[1]})", True, (200, 200, 200)), (x, y))


def draw_roi_selection_overlay(surface, roi_start, roi_current, color=(0, 255, 255)):
    """
    Draw ROI selection rectangle during drawing.
    
    Args:
        surface: Pygame surface
        roi_start: (x, y) start corner
        roi_current: (x, y) current corner
        color: RGB color
    """
    if roi_start is None or roi_current is None:
        return
    
    x1, y1 = roi_start
    x2, y2 = roi_current
    
    # Ensure proper ordering
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    if w > 0 and h > 0:
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, color, rect, 2)
        
        # Draw diagonal lines to indicate selection area
        pygame.draw.line(surface, color, (x, y), (x + w, y + h), 1)
        pygame.draw.line(surface, color, (x + w, y), (x, y + h), 1)


def draw_instructions(surface, font, instructions, pos, color=(200, 200, 200)):
    """
    Draw instruction text.
    
    Args:
        surface: Pygame surface
        font: Pygame font
        instructions: List of instruction strings
        pos: (x, y) position
        color: RGB color
    """
    x, y = pos
    for text in instructions:
        surface.blit(font.render(text, True, color), (x, y))
        y += 20

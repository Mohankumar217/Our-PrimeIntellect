"""
3️⃣ Video Perception Layer
Purpose: Infer state-like information ONLY from frames.
NO ACCESS TO: game grid, coordinates, tile types, or environment internals.
ALLOWED: Color detection, bounding boxes, frame differencing, pixel heuristics.
"""
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional


class VideoPerceptionLayer:
    """Infers observations from video frames using only visual analysis."""
    
    # Expected colors (RGB)
    AGENT_COLOR = (0, 0, 255)      # Blue
    HOLE_COLOR = (255, 0, 0)       # Red
    GOAL_COLOR = (0, 255, 0)       # Green
    SAFE_COLOR = (255, 255, 255)   # White
    
    def __init__(self, cell_size=100, grid_rows=4, grid_cols=4):
        """
        Args:
            cell_size: Expected pixel size of each grid cell
            grid_rows: Expected number of rows in grid
            grid_cols: Expected number of columns in grid
        """
        self.cell_size = cell_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        self.previous_frame = None
    
    def _color_match(self, pixel, target_color, tolerance=30):
        """Check if pixel matches target color within tolerance."""
        return all(abs(pixel[i] - target_color[i]) <= tolerance for i in range(3))
    
    def _find_agent_position(self, frame: Image.Image) -> Optional[Tuple[int, int]]:
        """
        Detect agent position by finding blue pixels.
        
        Args:
            frame: PIL Image
        
        Returns:
            (row, col) tuple or None if not detected
        """
        frame_array = np.array(frame)
        
        # Find all blue pixels (agent color)
        blue_mask = np.all(np.abs(frame_array - self.AGENT_COLOR) <= 30, axis=2)
        
        if not blue_mask.any():
            return None
        
        # Find centroid of blue pixels
        blue_pixels = np.argwhere(blue_mask)
        centroid_y, centroid_x = blue_pixels.mean(axis=0)
        
        # Convert pixel position to grid position
        row = int(centroid_y / self.cell_size)
        col = int(centroid_x / self.cell_size)
        
        return (row, col)
    
    def _find_goal_position(self, frame: Image.Image) -> Optional[Tuple[int, int]]:
        """
        Detect goal position by finding green pixels.
        
        Args:
            frame: PIL Image
        
        Returns:
            (row, col) tuple or None if not detected
        """
        frame_array = np.array(frame)
        
        # Find all green pixels (goal color)
        green_mask = np.all(np.abs(frame_array - self.GOAL_COLOR) <= 30, axis=2)
        
        if not green_mask.any():
            return None
        
        # Find centroid of green pixels
        green_pixels = np.argwhere(green_mask)
        centroid_y, centroid_x = green_pixels.mean(axis=0)
        
        # Convert pixel position to grid position
        row = int(centroid_y / self.cell_size)
        col = int(centroid_x / self.cell_size)
        
        return (row, col)
    
    def _detect_nearby_holes(self, frame: Image.Image, agent_pos: Tuple[int, int]) -> bool:
        """
        Check if there are red (hole) pixels near the agent.
        
        Args:
            frame: PIL Image
            agent_pos: (row, col) of agent
        
        Returns:
            True if holes detected nearby
        """
        frame_array = np.array(frame)
        
        # Check adjacent cells for red color
        agent_row, agent_col = agent_pos
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                check_row = agent_row + dr
                check_col = agent_col + dc
                
                if 0 <= check_row < self.grid_rows and 0 <= check_col < self.grid_cols:
                    # Sample center pixel of that cell
                    y = check_row * self.cell_size + self.cell_size // 2
                    x = check_col * self.cell_size + self.cell_size // 2
                    
                    pixel = tuple(frame_array[y, x])
                    if self._color_match(pixel, self.HOLE_COLOR):
                        return True
        
        return False
    
    def _calculate_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """
        Calculate relative direction between two positions.
        
        Args:
            from_pos: (row, col) starting position
            to_pos: (row, col) target position
        
        Returns:
            Direction string like "down-right", "up", "same"
        """
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        
        if row_diff == 0 and col_diff == 0:
            return "same"
        
        vertical = ""
        if row_diff > 0:
            vertical = "down"
        elif row_diff < 0:
            vertical = "up"
        
        horizontal = ""
        if col_diff > 0:
            horizontal = "right"
        elif col_diff < 0:
            horizontal = "left"
        
        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        elif vertical:
            return vertical
        else:
            return horizontal
    
    def perceive(self, frame: Image.Image) -> Dict:
        """
        Analyze frame and extract observations.
        
        Args:
            frame: Current frame image
        
        Returns:
            Dictionary with inferred observations
        """
        agent_pos = self._find_agent_position(frame)
        goal_pos = self._find_goal_position(frame)
        
        observation = {
            "agent_visible": agent_pos is not None,
            "agent_position_inferred": agent_pos,  # For internal use only
            "goal_direction": None,
            "danger_nearby": False,
            "movement_detected": False
        }
        
        if agent_pos and goal_pos:
            observation["goal_direction"] = self._calculate_direction(agent_pos, goal_pos)
        
        if agent_pos:
            observation["danger_nearby"] = self._detect_nearby_holes(frame, agent_pos)
        
        # Detect movement by comparing with previous frame
        if self.previous_frame is not None and agent_pos:
            prev_agent_pos = self._find_agent_position(self.previous_frame)
            if prev_agent_pos and prev_agent_pos != agent_pos:
                observation["movement_detected"] = True
        
        # Store for next comparison
        self.previous_frame = frame.copy()
        
        return observation
    
    def reset(self):
        """Reset perception state for new episode."""
        self.previous_frame = None

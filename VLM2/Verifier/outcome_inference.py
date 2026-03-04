"""
5️⃣ Outcome Inference Module
Purpose: Detect success or failure from visuals.
NO ACCESS TO: reward function or environment done flag.
ONLY DOES: analyze frames for terminal states and progress.
"""
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional


class OutcomeInferenceModule:
    """Infers episode outcomes from visual observations."""
    
    AGENT_COLOR = (0, 0, 255)    # Blue
    HOLE_COLOR = (255, 0, 0)     # Red
    GOAL_COLOR = (0, 255, 0)     # Green
    
    def __init__(self, cell_size=100):
        """
        Args:
            cell_size: Pixel size of each grid cell
        """
        self.cell_size = cell_size
    
    def _find_agent_position(self, frame: Image.Image) -> Optional[Tuple[int, int]]:
        """Find agent position by detecting blue pixels."""
        frame_array = np.array(frame)
        blue_mask = np.all(np.abs(frame_array - self.AGENT_COLOR) <= 30, axis=2)
        
        if not blue_mask.any():
            return None
        
        blue_pixels = np.argwhere(blue_mask)
        centroid_y, centroid_x = blue_pixels.mean(axis=0)
        
        row = int(centroid_y / self.cell_size)
        col = int(centroid_x / self.cell_size)
        
        return (row, col)
    
    def _get_cell_color(self, frame: Image.Image, row: int, col: int) -> Tuple[int, int, int]:
        """
        Get the dominant background color of a grid cell.
        
        Args:
            frame: PIL Image
            row: Grid row
            col: Grid column
        
        Returns:
            RGB color tuple
        """
        frame_array = np.array(frame)
        
        # Sample the cell's background (avoid the agent overlay)
        y = row * self.cell_size + 5  # Top-left corner of cell
        x = col * self.cell_size + 5
        
        return tuple(frame_array[y, x])
    
    def _color_match(self, color1, color2, tolerance=30):
        """Check if two colors match within tolerance."""
        return all(abs(color1[i] - color2[i]) <= tolerance for i in range(3))
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _find_goal_position(self, frame: Image.Image) -> Optional[Tuple[int, int]]:
        """Find goal position by detecting green pixels."""
        frame_array = np.array(frame)
        green_mask = np.all(np.abs(frame_array - self.GOAL_COLOR) <= 30, axis=2)
        
        if not green_mask.any():
            return None
        
        green_pixels = np.argwhere(green_mask)
        centroid_y, centroid_x = green_pixels.mean(axis=0)
        
        row = int(centroid_y / self.cell_size)
        col = int(centroid_x / self.cell_size)
        
        return (row, col)
    
    def infer_outcome(self, frame: Image.Image, 
                     prev_frame: Optional[Image.Image] = None,
                     max_steps_reached: bool = False) -> Dict:
        """
        Analyze current frame to detect terminal states and progress.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (for progress calculation)
            max_steps_reached: Whether episode hit step limit
        
        Returns:
            Dictionary with outcome information
        """
        agent_pos = self._find_agent_position(frame)
        
        outcome = {
            "terminal": False,
            "progress": "neutral",
            "outcome": "ongoing"
        }
        
        if agent_pos is None:
            # Agent disappeared - likely fell in hole
            outcome["terminal"] = True
            outcome["outcome"] = "failure"
            return outcome
        
        # Check if agent is on red tile (hole)
        cell_color = self._get_cell_color(frame, agent_pos[0], agent_pos[1])
        if self._color_match(cell_color, self.HOLE_COLOR):
            outcome["terminal"] = True
            outcome["outcome"] = "failure"
            return outcome
        
        # Check if agent is on green tile (goal)
        if self._color_match(cell_color, self.GOAL_COLOR):
            outcome["terminal"] = True
            outcome["outcome"] = "success"
            return outcome
        
        # Check if max steps reached
        if max_steps_reached:
            outcome["terminal"] = True
            outcome["outcome"] = "timeout"
            return outcome
        
        # Calculate progress if previous frame available
        if prev_frame is not None:
            goal_pos = self._find_goal_position(frame)
            prev_agent_pos = self._find_agent_position(prev_frame)
            
            if goal_pos and prev_agent_pos:
                prev_distance = self._calculate_distance(prev_agent_pos, goal_pos)
                curr_distance = self._calculate_distance(agent_pos, goal_pos)
                
                if curr_distance < prev_distance:
                    outcome["progress"] = "positive"
                elif curr_distance > prev_distance:
                    outcome["progress"] = "negative"
                else:
                    outcome["progress"] = "neutral"
        
        return outcome

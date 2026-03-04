"""
4️⃣ Action Inference Module
Purpose: Infer what action was taken by observing frame changes.
NO ACCESS TO: agent decision logic or action commands.
ONLY DOES: compare frames and infer movement direction.
"""
import numpy as np
from PIL import Image
from typing import Optional


class ActionInferenceModule:
    """Infers actions from frame-to-frame changes."""
    
    ACTIONS = {
        'LEFT': 0,
        'DOWN': 1,
        'RIGHT': 2,
        'UP': 3,
        'STAY': 4  # No movement or invalid action
    }
    
    def __init__(self, cell_size=100):
        """
        Args:
            cell_size: Pixel size of each grid cell
        """
        self.cell_size = cell_size
    
    def _find_agent_position(self, frame: Image.Image) -> Optional[tuple]:
        """
        Find agent position in frame by detecting blue pixels.
        
        Args:
            frame: PIL Image
        
        Returns:
            (row, col) or None
        """
        frame_array = np.array(frame)
        agent_color = (0, 0, 255)
        
        blue_mask = np.all(np.abs(frame_array - agent_color) <= 30, axis=2)
        
        if not blue_mask.any():
            return None
        
        blue_pixels = np.argwhere(blue_mask)
        centroid_y, centroid_x = blue_pixels.mean(axis=0)
        
        row = int(centroid_y / self.cell_size)
        col = int(centroid_x / self.cell_size)
        
        return (row, col)
    
    def infer_action(self, frame_before: Image.Image, frame_after: Image.Image) -> str:
        """
        Infer which action was taken between two frames.
        
        Args:
            frame_before: Frame at time t
            frame_after: Frame at time t+1
        
        Returns:
            Action name: 'LEFT', 'RIGHT', 'UP', 'DOWN', or 'STAY'
        """
        pos_before = self._find_agent_position(frame_before)
        pos_after = self._find_agent_position(frame_after)
        
        if pos_before is None or pos_after is None:
            return 'STAY'
        
        row_diff = pos_after[0] - pos_before[0]
        col_diff = pos_after[1] - pos_before[1]
        
        # Determine action from position change
        if row_diff == 0 and col_diff == 0:
            return 'STAY'
        elif row_diff == 0 and col_diff == -1:
            return 'LEFT'
        elif row_diff == 0 and col_diff == 1:
            return 'RIGHT'
        elif row_diff == -1 and col_diff == 0:
            return 'UP'
        elif row_diff == 1 and col_diff == 0:
            return 'DOWN'
        else:
            # Unexpected movement (shouldn't happen in FrozenLake)
            return 'STAY'
    
    def get_action_id(self, action_name: str) -> int:
        """Convert action name to numeric ID."""
        return self.ACTIONS.get(action_name, 4)

"""
Video-Based FrozenLake Environment
This is the GAME ENGINE - it manages the actual FrozenLake game state.
The modules above do NOT have access to this.
"""
import numpy as np
from typing import Tuple, Optional


class FrozenLakeGame:
    """
    Internal game engine for FrozenLake.
    This is HIDDEN from the video-based learning system.
    """
    
    # Action mappings
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    def __init__(self, map_desc=None, is_slippery=False):
        """
        Args:
            map_desc: List of strings describing the map
            is_slippery: Whether movement is stochastic
        """
        if map_desc is None:
            map_desc = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
            ]
        
        self.map_desc = map_desc
        self.nrows = len(map_desc)
        self.ncols = len(map_desc[0])
        self.is_slippery = is_slippery
        
        # Find start and goal positions
        self.start_pos = None
        self.goal_pos = None
        
        for r in range(self.nrows):
            for c in range(self.ncols):
                if map_desc[r][c] == 'S':
                    self.start_pos = (r, c)
                elif map_desc[r][c] == 'G':
                    self.goal_pos = (r, c)
        
        self.agent_pos = None
        self.done = False
    
    def reset(self) -> Tuple[int, int]:
        """Reset game to start position."""
        self.agent_pos = self.start_pos
        self.done = False
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], bool]:
        """
        Execute action and return new position and done flag.
        
        Args:
            action: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
        
        Returns:
            (new_position, done)
        """
        if self.done:
            return self.agent_pos, True
        
        row, col = self.agent_pos
        
        # Determine intended direction
        if action == self.LEFT:
            new_col = max(col - 1, 0)
            new_row = row
        elif action == self.RIGHT:
            new_col = min(col + 1, self.ncols - 1)
            new_row = row
        elif action == self.UP:
            new_row = max(row - 1, 0)
            new_col = col
        elif action == self.DOWN:
            new_row = min(row + 1, self.nrows - 1)
            new_col = col
        else:
            new_row, new_col = row, col
        
        # Update position
        self.agent_pos = (new_row, new_col)
        
        # Check terminal conditions
        tile = self.map_desc[new_row][new_col]
        if tile == 'H':  # Hole
            self.done = True
        elif tile == 'G':  # Goal
            self.done = True
        
        return self.agent_pos, self.done

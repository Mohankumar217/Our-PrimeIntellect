"""
1️⃣ FrozenLake Video Renderer
Purpose: Convert each environment step into a visual frame.
NO ACCESS TO: game logic, reward function, or agent decision making.
ONLY DOES: render grid state to image frames.
"""
import numpy as np
from PIL import Image, ImageDraw
import os


class FrozenLakeVideoRenderer:
    """Renders FrozenLake grid as visual frames."""
    
    # Color coding (RGB)
    COLORS = {
        'S': (255, 255, 255),  # Safe tile = White
        'F': (255, 255, 255),  # Frozen tile = White  
        'H': (255, 0, 0),      # Hole = Red
        'G': (0, 255, 0),      # Goal = Green
        'agent': (0, 0, 255)   # Agent = Blue
    }
    
    def __init__(self, map_desc, cell_size=100):
        """
        Args:
            map_desc: List of strings describing the grid layout
            cell_size: Pixel size of each grid cell
        """
        self.map_desc = map_desc
        self.rows = len(map_desc)
        self.cols = len(map_desc[0])
        self.cell_size = cell_size
        
        # Frame storage for current episode
        self.episode_frames = []
        self.frame_count = 0
        
    def render_frame(self, agent_row, agent_col):
        """
        Render a single frame showing the grid and agent position.
        
        Args:
            agent_row: Agent's row position
            agent_col: Agent's column position
        
        Returns:
            PIL Image object
        """
        # Create image
        width = self.cols * self.cell_size
        height = self.rows * self.cell_size
        img = Image.new('RGB', (width, height), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw grid tiles
        for r in range(self.rows):
            for c in range(self.cols):
                tile = self.map_desc[r][c]
                color = self.COLORS.get(tile, (255, 255, 255))
                
                x0 = c * self.cell_size
                y0 = r * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                
                # Draw tile background
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        
        # Draw agent as a smaller square on top of the tile
        agent_x0 = agent_col * self.cell_size + self.cell_size // 4
        agent_y0 = agent_row * self.cell_size + self.cell_size // 4
        agent_x1 = agent_x0 + self.cell_size // 2
        agent_y1 = agent_y0 + self.cell_size // 2
        
        draw.rectangle([agent_x0, agent_y0, agent_x1, agent_y1], 
                      fill=self.COLORS['agent'], 
                      outline=(0, 0, 0))
        
        return img
    
    def add_frame(self, agent_row, agent_col, save_to_disk=False, output_dir='frames'):
        """
        Create and store a frame for the current timestep.
        
        Args:
            agent_row: Current agent row
            agent_col: Current agent column
            save_to_disk: Whether to save frame as PNG
            output_dir: Directory to save frames
        
        Returns:
            The rendered frame image
        """
        frame = self.render_frame(agent_row, agent_col)
        self.episode_frames.append(frame)
        
        if save_to_disk:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f'frame_{self.frame_count:04d}.png')
            frame.save(filename)
        
        self.frame_count += 1
        return frame
    
    def get_frames(self):
        """Return all frames from current episode."""
        return self.episode_frames
    
    def reset(self):
        """Clear frames for new episode."""
        self.episode_frames = []
        self.frame_count = 0


# Default 4x4 FrozenLake map
DEFAULT_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

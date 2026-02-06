from PIL import Image, ImageDraw

class FrozenLakeRenderer:
    """
    Renders the FrozenLake world state into a visual image.
    Strictly visual - no text labels describing the state.
    """
    def __init__(self, tile_size=100):
        self.tile_size = tile_size
        
        # Colors (R, G, B)
        self.colors = {
            'S': (200, 200, 255), # Start (Light Blue)
            'F': (240, 255, 255), # Frozen (Very Light Cyan)
            'H': (20, 20, 20),    # Hole (Almost Black)
            'G': (255, 215, 0),   # Goal (Gold)
            'AGENT': (255, 50, 50), # Agent (Red)
            'GRID': (100, 100, 100) # Grid lines (Gray)
        }

    def render(self, world):
        """
        Generates an RGB image of the current world state.
        
        Args:
            world: The FrozenLakeWorld instance.
            
        Returns:
            PIL.Image: The rendered frame.
        """
        rows = world.rows
        cols = world.cols
        width = cols * self.tile_size
        height = rows * self.tile_size
        
        # Create blank image
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        
        # Draw Map
        for r in range(rows):
            for c in range(cols):
                tile_type = world.grid_map[r][c]
                color = self.colors.get(tile_type, (255, 255, 255))
                
                x1 = c * self.tile_size
                y1 = r * self.tile_size
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                
                # Draw tile background
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=self.colors['GRID'])
        
        # Draw Agent
        agent_r, agent_c = world.agent_pos
        padding = self.tile_size // 4
        
        ax1 = agent_c * self.tile_size + padding
        ay1 = agent_r * self.tile_size + padding
        ax2 = (agent_c + 1) * self.tile_size - padding
        ay2 = (agent_r + 1) * self.tile_size - padding
        
        draw.ellipse([ax1, ay1, ax2, ay2], fill=self.colors['AGENT'], outline=(0, 0, 0))
        
        return image

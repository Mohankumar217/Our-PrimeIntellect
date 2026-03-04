"""
2️⃣ Episode Video Builder
Purpose: Convert frames → video.
NO ACCESS TO: game state, agent logic, or environment internals.
ONLY DOES: stitch image frames into MP4 video.
"""
import cv2
import numpy as np
import os
from typing import List
from PIL import Image


class EpisodeVideoBuilder:
    """Stitches frames into episode videos."""
    
    def __init__(self, fps=2, output_dir='videos'):
        """
        Args:
            fps: Frames per second for output video
            output_dir: Directory to save videos
        """
        self.fps = fps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def build_video(self, frames: List[Image.Image], episode_num: int, filename=None):
        """
        Stitch frames into a video file.
        
        Args:
            frames: List of PIL Image frames
            episode_num: Episode number for naming
            filename: Optional custom filename
        
        Returns:
            Path to saved video file
        """
        if not frames:
            raise ValueError("No frames provided to build video")
        
        # Determine output filename
        if filename is None:
            filename = f'episode_{episode_num:04d}.mp4'
        
        video_path = os.path.join(self.output_dir, filename)
        
        # Get dimensions from first frame
        width, height = frames[0].size
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        # Write each frame
        for frame in frames:
            # Convert PIL Image to OpenCV format (BGR)
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        
        return video_path
    
    def build_from_image_files(self, image_dir: str, episode_num: int):
        """
        Build video from saved PNG files.
        
        Args:
            image_dir: Directory containing frame_*.png files
            episode_num: Episode number for naming
        
        Returns:
            Path to saved video file
        """
        # Load all frame images
        frame_files = sorted([f for f in os.listdir(image_dir) if f.startswith('frame_')])
        
        frames = []
        for frame_file in frame_files:
            img_path = os.path.join(image_dir, frame_file)
            img = Image.open(img_path)
            frames.append(img)
        
        return self.build_video(frames, episode_num)

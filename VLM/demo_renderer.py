import sys
import os
from PIL import Image

# Ensure imports work from VLM root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from World.frozenlake_world import FrozenLakeWorld
from World.frozenlake_renderer import FrozenLakeRenderer

def run_demo():
    print("Initializing World and Renderer...")
    world = FrozenLakeWorld()
    renderer = FrozenLakeRenderer()

    # Reset
    print("Resetting World...")
    world.reset()
    
    # Save initial frame
    img = renderer.render(world)
    img_path = os.path.join(os.path.dirname(__file__), "demo_frame_0_start.png")
    img.save(img_path)
    print(f"Saved {img_path}")

    # Step LEFT (Hit Wall)
    print("Action: LEFT")
    world.step("LEFT")
    img = renderer.render(world)
    img.save(os.path.join(os.path.dirname(__file__), "demo_frame_1_left_wall.png"))

    # Step RIGHT (Safe)
    print("Action: RIGHT")
    world.step("RIGHT")
    img = renderer.render(world)
    img.save(os.path.join(os.path.dirname(__file__), "demo_frame_2_right.png"))
    
    # Step DOWN (Hole)
    print("Action: DOWN (Into Hole!)")
    world.step("DOWN")
    img = renderer.render(world)
    img.save(os.path.join(os.path.dirname(__file__), "demo_frame_3_hole_gameover.png"))

    print("Demo execution complete. Check the .png files in this folder.")

if __name__ == "__main__":
    run_demo()

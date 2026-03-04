"""
Runnable Demo: Video-Based FrozenLake Learning

This demonstrates the complete system where an agent learns
ONLY from video frames, without accessing game state.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wrapper.video_environment import VideoBasedEnvironment
import random
from PIL import Image


def simple_heuristic_agent(frame: Image.Image, observation: dict, memories: list) -> int:
    """
    A simple rule-based agent that decides actions based on:
    1. Visual observations from the frame
    2. Past experiences from memory
    
    NO ACCESS TO: game state, coordinates, or reward function
    ONLY USES: frame pixels and memory
    
    Args:
        frame: Current frame image
        observation: Perception output (goal_direction, danger_nearby, etc.)
        memories: Relevant past experiences
    
    Returns:
        Action ID (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
    """
    # Action mapping
    ACTIONS = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3}
    
    # Check if we have successful memories to imitate
    if memories:
        for memory in memories:
            if memory['outcome'] == 'success':
                # Try to extract action from memory lesson
                # (This is simplistic - a real agent would be more sophisticated)
                if 'RIGHT' in memory['action']:
                    return ACTIONS['RIGHT']
                elif 'DOWN' in memory['action']:
                    return ACTIONS['DOWN']
                elif 'UP' in memory['action']:
                    return ACTIONS['UP']
                elif 'LEFT' in memory['action']:
                    return ACTIONS['LEFT']
    
    # Avoid danger
    if observation.get('danger_nearby'):
        # If danger nearby, prefer moving toward goal
        goal_dir = observation.get('goal_direction', '')
        
        if 'down' in goal_dir and 'right' in goal_dir:
            return random.choice([ACTIONS['DOWN'], ACTIONS['RIGHT']])
        elif 'down' in goal_dir:
            return ACTIONS['DOWN']
        elif 'right' in goal_dir:
            return ACTIONS['RIGHT']
        elif 'up' in goal_dir:
            return ACTIONS['UP']
        elif 'left' in goal_dir:
            return ACTIONS['LEFT']
    
    # Otherwise, move toward goal
    goal_dir = observation.get('goal_direction', '')
    
    if 'down' in goal_dir:
        return ACTIONS['DOWN']
    elif 'right' in goal_dir:
        return ACTIONS['RIGHT']
    elif 'up' in goal_dir:
        return ACTIONS['UP']
    elif 'left' in goal_dir:
        return ACTIONS['LEFT']
    
    # Random if no clear direction
    return random.choice([ACTIONS['LEFT'], ACTIONS['DOWN'], ACTIONS['RIGHT'], ACTIONS['UP']])


def run_demo(num_episodes=10):
    """
    Run the video-based learning demo.
    
    Args:
        num_episodes: Number of episodes to run
    """
    print("=" * 60)
    print("VIDEO-BASED FROZENLAKE LEARNING DEMO")
    print("=" * 60)
    print()
    print("Constraints:")
    print("  ❌ No symbolic state (x,y coordinates, grid arrays)")
    print("  ❌ No direct reward from environment")
    print("  ❌ No access to environment internals")
    print()
    print("  ✅ Agent sees ONLY video frames")
    print("  ✅ Learning signals derived from video")
    print("  ✅ Memory based on visual experiences")
    print()
    print("=" * 60)
    print()
    
    # Create environment
    env = VideoBasedEnvironment()
    
    # Run episodes
    successes = 0
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}...")
        
        result = env.run_episode_with_agent(simple_heuristic_agent, max_steps=50)
        
        print(f"  Outcome: {result['final_outcome']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Video: {result['video_path']}")
        
        if result['final_outcome'] == 'success':
            successes += 1
        
        # Show memory stats
        stats = env.memory.get_statistics()
        print(f"  Memory: {stats['total_experiences']} experiences "
              f"({stats['successes']} successes, {stats['failures']} failures)")
        print()
    
    # Final statistics
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    
    # Show final memory state
    print()
    print("Memory Statistics:")
    stats = env.memory.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save memory
    memory_path = "trajectory_memory.json"
    env.memory.save_to_file(memory_path)
    print(f"\nMemory saved to: {memory_path}")
    
    # Show some learned experiences
    print()
    print("Sample Learned Experiences:")
    print("-" * 60)
    for i, exp in enumerate(env.memory.trajectories[:5]):
        print(f"{i+1}. Situation: {exp['situation']}")
        print(f"   Action: {exp['action']}")
        print(f"   Outcome: {exp['outcome']}")
        print(f"   Lesson: {exp['lesson']}")
        print()
    
    print("=" * 60)
    print("Demo complete! Videos saved to 'videos/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo(num_episodes=10)

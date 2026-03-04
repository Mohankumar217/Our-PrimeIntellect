"""
7️⃣ Agent Interaction Loop
Purpose: Orchestrate the video-based learning environment.
The agent receives ONLY frames, never the internal game state.
"""
from World.video_renderer import FrozenLakeVideoRenderer, DEFAULT_MAP
from World.video_builder import EpisodeVideoBuilder
from World.frozenlake_game import FrozenLakeGame
from Wrapper.video_perception import VideoPerceptionLayer
from Verifier.action_inference import ActionInferenceModule
from Verifier.outcome_inference import OutcomeInferenceModule
from Memory.trajectory_memory import TrajectoryMemory

from typing import Dict, List, Callable
from PIL import Image


class VideoBasedEnvironment:
    """
    The main environment that agents interact with.
    Agents see ONLY video frames, never internal state.
    """
    
    def __init__(self, map_desc=None, cell_size=100, max_steps=50):
        """
        Args:
            map_desc: Grid map description
            cell_size: Pixel size of each cell
            max_steps: Maximum steps per episode
        """
        if map_desc is None:
            map_desc = DEFAULT_MAP
        
        self.map_desc = map_desc
        self.cell_size = cell_size
        self.max_steps = max_steps
        
        # Internal game engine (HIDDEN from agent)
        self.game = FrozenLakeGame(map_desc=map_desc)
        
        # Video and perception modules
        self.renderer = FrozenLakeVideoRenderer(map_desc, cell_size)
        self.video_builder = EpisodeVideoBuilder(fps=2)
        self.perception = VideoPerceptionLayer(cell_size, len(map_desc), len(map_desc[0]))
        self.action_inference = ActionInferenceModule(cell_size)
        self.outcome_inference = OutcomeInferenceModule(cell_size)
        
        # Memory system
        self.memory = TrajectoryMemory(max_size=100, top_k=20)
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.previous_frame = None
    
    def reset(self) -> Image.Image:
        """
        Reset environment for new episode.
        
        Returns:
            Initial frame image
        """
        # Reset internal game
        agent_pos = self.game.reset()
        
        # Reset modules
        self.renderer.reset()
        self.perception.reset()
        
        # Render initial frame
        initial_frame = self.renderer.add_frame(agent_pos[0], agent_pos[1])
        
        self.current_step = 0
        self.previous_frame = initial_frame
        
        return initial_frame
    
    def step(self, action: int) -> Dict:
        """
        Execute one step in the environment.
        
        Args:
            action: Action ID (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
        
        Returns:
            Dictionary containing:
            - 'frame': Current frame image
            - 'observation': Perception output
            - 'outcome': Outcome inference
            - 'done': Whether episode is terminal
        """
        # Execute action in internal game
        new_pos, game_done = self.game.step(action)
        
        # Render new frame
        current_frame = self.renderer.add_frame(new_pos[0], new_pos[1])
        
        # Perceive from frame
        observation = self.perception.perceive(current_frame)
        
        # Infer outcome
        max_steps_reached = (self.current_step + 1 >= self.max_steps)
        outcome = self.outcome_inference.infer_outcome(
            current_frame, 
            self.previous_frame,
            max_steps_reached
        )
        
        self.current_step += 1
        self.previous_frame = current_frame
        
        return {
            'frame': current_frame,
            'observation': observation,
            'outcome': outcome,
            'done': outcome['terminal'] or game_done
        }
    
    def finish_episode(self) -> str:
        """
        Finish current episode and create video.
        
        Returns:
            Path to episode video
        """
        frames = self.renderer.get_frames()
        video_path = self.video_builder.build_video(frames, self.current_episode)
        
        self.current_episode += 1
        
        return video_path
    
    def get_observation_summary(self, observation: Dict) -> str:
        """
        Convert observation dict to natural language summary.
        
        Args:
            observation: Perception output
        
        Returns:
            Natural language description
        """
        parts = []
        
        if observation.get('agent_visible'):
            parts.append("Agent visible")
        
        if observation.get('goal_direction'):
            parts.append(f"Goal is {observation['goal_direction']}")
        
        if observation.get('danger_nearby'):
            parts.append("Danger nearby")
        
        if observation.get('movement_detected'):
            parts.append("Movement detected")
        
        return ", ".join(parts) if parts else "No clear observation"
    
    def run_episode_with_agent(self, agent_fn: Callable, max_steps=None) -> Dict:
        """
        Run a full episode with an agent function.
        
        Args:
            agent_fn: Function that takes (frame, observation, memory) and returns action
            max_steps: Override max steps
        
        Returns:
            Episode summary dictionary
        """
        if max_steps is None:
            max_steps = self.max_steps
        
        # Reset environment
        frame = self.reset()
        done = False
        
        episode_data = {
            'frames': [],
            'actions': [],
            'observations': [],
            'outcomes': []
        }
        
        step_count = 0
        
        while not done and step_count < max_steps:
            # Get observation from frame
            observation = self.perception.perceive(frame)
            obs_summary = self.get_observation_summary(observation)
            
            # Retrieve relevant memories
            relevant_memories = self.memory.retrieve_relevant(obs_summary, k=3)
            
            # Agent chooses action based on frame and memory
            action = agent_fn(frame, observation, relevant_memories)
            
            # Execute action
            step_result = self.step(action)
            
            # Store data
            episode_data['frames'].append(step_result['frame'])
            episode_data['actions'].append(action)
            episode_data['observations'].append(step_result['observation'])
            episode_data['outcomes'].append(step_result['outcome'])
            
            frame = step_result['frame']
            done = step_result['done']
            step_count += 1
        
        # Create episode video
        video_path = self.finish_episode()
        
        # Determine final outcome
        final_outcome = episode_data['outcomes'][-1] if episode_data['outcomes'] else {'outcome': 'unknown'}
        
        # Store valuable experience in memory
        if final_outcome['outcome'] == 'success':
            self.memory.add_experience(
                situation=self.get_observation_summary(episode_data['observations'][-1]),
                action=f"Action {episode_data['actions'][-1]}",
                outcome='success',
                lesson="Successfully reached the goal"
            )
        elif final_outcome['outcome'] == 'failure' and episode_data['observations']:
            self.memory.add_experience(
                situation=self.get_observation_summary(episode_data['observations'][-1]),
                action=f"Action {episode_data['actions'][-1]}",
                outcome='failure',
                lesson="Avoid this action in this situation"
            )
        
        return {
            'video_path': video_path,
            'final_outcome': final_outcome['outcome'],
            'steps': step_count,
            'episode_data': episode_data
        }

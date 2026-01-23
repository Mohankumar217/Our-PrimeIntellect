import json
import os
from typing import List, Dict, Any

class TrajectoryMemory:
    """
    Manages persistent storage of successful episodes.
    Retains only the Top-K episodes based on a fitness score.
    """
    def __init__(self, filepath: str = "memory.json", k: int = 5):
        self.filepath = filepath
        self.k = k
        self.episodes: List[Dict[str, Any]] = []
        self._load_memory()

    def _load_memory(self):
        """Loads memory from JSON file if it exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.episodes = json.load(f)
                # Re-sort just in case file was tampered with
                self.episodes.sort(key=lambda x: x.get('fitness', -float('inf')), reverse=True)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load memory from {self.filepath}. Starting fresh.")
                self.episodes = []

    def _save_memory(self):
        """Saves current episodes to JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.episodes, f, indent=2)
        except IOError as e:
            print(f"Error saving memory: {e}")

    def calculate_fitness(self, score: float, steps: int) -> float:
        """
        Calculates fitness score.
        Formula: fitness = score - (0.05 * steps)
        Promotes high scores achieved in fewer steps.
        """
        return score - (0.05 * steps)

    def add_episode(self, episode_data: Dict[str, Any]):
        """
        Adds a new episode if it qualifies for Top-K.
        
        Args:
            episode_data: Dict containing:
                - trajectory: List of {state, action, response, outcome}
                - steps: int
                - score: float
                - final_outcome: str
        """
        # Calculate fitness
        fitness = self.calculate_fitness(episode_data['score'], episode_data['steps'])
        episode_data['fitness'] = fitness

        # Add to list
        self.episodes.append(episode_data)

        # Sort by fitness descending
        self.episodes.sort(key=lambda x: x['fitness'], reverse=True)

        # Keep top K
        if len(self.episodes) > self.k:
            self.episodes = self.episodes[:self.k]
        
        # Persist
        self._save_memory()

    def get_top_k(self) -> List[Dict[str, Any]]:
        """Returns the top K episodes."""
        return self.episodes

    def get_lessons(self) -> str:
        """
        Distills memory into a string of lessons for the prompt.
        """
        if not self.episodes:
            return "No previous successful strategies available."
        
        lessons = [f"--- Success (Fitness: {ep['fitness']:.2f}) ---"]
        for ep in self.episodes:
            # We can summarize the path taken
            path_summary = []
            for step in ep['trajectory']:
                # Extract coordinates from message if possible, or just action
                # Assuming simple string format for now based on log
                path_summary.append(f"{step.get('action', 'Unknown')}")
            
            lessons.append(f"Generic Result: {ep['final_outcome']} in {ep['steps']} steps.")
            # lessons.append(f"Path: {' -> '.join(path_summary)}") # Optional verbose path
        
        return "\n".join(lessons)

import json
import os
from typing import List, Dict, Any

class TrajectoryMemory:
    """
    Manages persistent storage of successful episodes.
    Retains only the Top-K episodes based on a fitness score.
    """
    def __init__(self, filepath: str = "memory.json"):
        """
        Initializes the Memory System.
        Now primarily a Q-Table store: State(Coords) -> {Action: Q-Value}
        """
        self.filepath = filepath
        self.q_table: Dict[str, Dict[str, float]] = {}
        self._load_memory()

    def _load_memory(self):
        """Loads Q-table from JSON file if it exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.q_table = json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load memory from {self.filepath}. Starting fresh.")
                self.q_table = {}

    def _save_memory(self):
        """Saves current Q-table to JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.q_table, f, indent=2, sort_keys=True)
        except IOError as e:
            print(f"Error saving memory: {e}")

    def get_q_values(self, state: tuple) -> Dict[str, float]:
        """
        Returns the Q-values for a given state.
        Returns default 0.0 for unseen actions.
        """
        state_key = str(state)
        if state_key not in self.q_table:
            return {"UP": 0.0, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 0.0}
        
        # Ensure all 4 actions are present
        values = self.q_table[state_key].copy()
        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if action not in values:
                values[action] = 0.0
        return values

    def update_step(self, state: tuple, action: str, reward: float, next_state: tuple, done: bool):
        """
        Performs a single Q-Learning update step.
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        """
        if action not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return

        alpha = 0.1 # Learning rate
        gamma = 0.9 # Discount factor
        
        state_key = str(state)
        # Initialize state if new
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Get current Q
        current_q = self.q_table[state_key].get(action, 0.0)
        
        # Get Max Next Q
        max_next_q = 0.0
        if not done:
            next_state_key = str(next_state)
            if next_state_key in self.q_table:
                # Max over existing actions (default 0 if empty)
                if self.q_table[next_state_key]:
                    max_next_q = max(self.q_table[next_state_key].values())
        
        # Calculate Target
        target = reward + gamma * max_next_q
        
        # Update
        new_q = current_q + alpha * (target - current_q)
        self.q_table[state_key][action] = round(new_q, 4) # Round for cleaner JSON
        
        self._save_memory()

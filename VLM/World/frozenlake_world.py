import copy

class FrozenLakeWorld:
    """
    A standalone, text-based FrozenLake world simulator.
    Independent of Gymnasium and any RL framework.
    """

    def __init__(self, grid_map=None):
        """
        Initialize the FrozenLake world.
        
        Args:
            grid_map (list[str], optional): The grid layout as a list of strings.
                                            Defaults to a standard 4x4 map.
        """
        if grid_map is None:
            self.grid_map = [
                "SFFF",
                "FHFF",
                "FFFH",
                "HFFG"
            ]
        else:
            self.grid_map = grid_map

        self.rows = len(self.grid_map)
        self.cols = len(self.grid_map[0])
        self.start_pos = self._find_start()
        self.agent_pos = self.start_pos
        self.terminated = False
        self.outcome = "ongoing" # ongoing, hole, goal

        # Define movement deltas (row, col)
        self.actions = {
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
            "UP": (-1, 0),
            "DOWN": (1, 0)
        }

    def _find_start(self):
        """Finds the starting position (S) in the grid."""
        for r, row in enumerate(self.grid_map):
            for c, char in enumerate(row):
                if char == 'S':
                    self.start_pos = (r, c)
                elif char == 'G':
                    self.goal_pos = (r, c)
        
        if not hasattr(self, 'start_pos'):
            raise ValueError("Grid must contain a Start 'S' tile.")
        if not hasattr(self, 'goal_pos'):
            raise ValueError("Grid must contain a Goal 'G' tile.")
        
        return self.start_pos

    def reset(self):
        """
        Resets the world to the starting state.

        Returns:
            dict: The initial observation dictionary.
        """
        self.agent_pos = self.start_pos
        self.terminated = False
        self.outcome = "ongoing"
        
        return self._get_observation("Game started. Good luck!")

    def step(self, action):
        """
        Executes a step in the world based on the given action.

        Args:
            action (str): The action to take ("LEFT", "RIGHT", "UP", "DOWN").
                          Case-insensitive.

        Returns:
            dict: A dictionary containing the new state, termination status, and outcome.
        """
        if self.terminated:
             return self._get_observation("Game is already over. Please reset.")

        action = action.upper()
        if action not in self.actions:
            return self._get_observation(f"Invalid action: {action}. Please choose LEFT, RIGHT, UP, or DOWN.")

        delta_r, delta_c = self.actions[action]
        current_r, current_c = self.agent_pos
        
        # Calculate tentative new position
        new_r = current_r + delta_r
        new_c = current_c + delta_c

        # Check boundaries
        if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
            self.agent_pos = (new_r, new_c)
            move_msg = f"You moved {action}."
        else:
            # Hit a wall, stay in place
            move_msg = f"You tried to move {action} but hit a wall."
            # Agent position does not change

        # Check tile type at current position
        r, c = self.agent_pos
        tile_type = self.grid_map[r][c]
        
        message = ""
        
        if tile_type == 'H':
            self.terminated = True
            self.outcome = "hole"
            message = f"{move_msg} You fell into a hole. Game over."
        elif tile_type == 'G':
            self.terminated = True
            self.outcome = "goal"
            message = f"{move_msg} You reached the goal! Success."
        else:
            self.terminated = False
            self.outcome = "ongoing"
            message = f"{move_msg} Current tile: {tile_type}."

        return self._get_observation(message)

    def _get_observation(self, message):
        """
        Helper to construct the observation dictionary.

        Args:
            message (str): A human-readable message describing the event.

        Returns:
            dict: The state observation.
        """
        r, c = self.agent_pos
        tile_type = self.grid_map[r][c]
        
        return {
            "position": (r, c),
            "goal_pos": self.goal_pos,
            "tile": tile_type,
            "terminated": self.terminated,
            "outcome": self.outcome,
            "message": message
        }

"""
6️⃣ Trajectory Memory (Video-Driven)
Purpose: Store only useful experiences derived from video observations.
NO ACCESS TO: game state or symbolic rewards.
ONLY DOES: store and retrieve video-based experiences.
"""
import json
from typing import List, Dict, Optional
from collections import deque


class TrajectoryMemory:
    """Stores video-based learning experiences."""
    
    def __init__(self, max_size=100, top_k=20):
        """
        Args:
            max_size: Maximum number of trajectories to store
            top_k: Number of most informative trajectories to keep
        """
        self.max_size = max_size
        self.top_k = top_k
        
        self.trajectories = []
        self.success_count = 0
        self.failure_count = 0
        
        # Track unique failure patterns to avoid duplicates
        self.seen_failures = set()
    
    def _is_new_failure(self, situation: str, action: str) -> bool:
        """Check if this failure pattern is new."""
        pattern = f"{situation}|{action}"
        if pattern in self.seen_failures:
            return False
        self.seen_failures.add(pattern)
        return True
    
    def add_experience(self, situation: str, action: str, outcome: str, lesson: str):
        """
        Add a new experience to memory.
        
        Args:
            situation: Visual observation summary
            action: Inferred action taken
            outcome: 'success', 'failure', or 'ongoing'
            lesson: Natural language lesson learned
        """
        experience = {
            "situation": situation,
            "action": action,
            "outcome": outcome,
            "lesson": lesson
        }
        
        # Selection rules:
        # - Always store SUCCESS
        # - Store FAILURE only if it teaches something new
        should_store = False
        
        if outcome == "success":
            should_store = True
            self.success_count += 1
        elif outcome == "failure":
            if self._is_new_failure(situation, action):
                should_store = True
                self.failure_count += 1
        
        if should_store:
            self.trajectories.append(experience)
            
            # Maintain maximum size
            if len(self.trajectories) > self.max_size:
                self._prune_memory()
    
    def _calculate_informativeness(self, experience: Dict) -> float:
        """
        Score how informative an experience is.
        Higher score = more valuable.
        
        Heuristic:
        - Success experiences are highly valuable
        - Novel failures are valuable
        - Common failures are less valuable
        """
        if experience["outcome"] == "success":
            return 10.0
        elif experience["outcome"] == "failure":
            # Failures are valuable but less than success
            return 5.0
        else:
            return 1.0
    
    def _prune_memory(self):
        """Keep only top-K most informative trajectories."""
        # Score all experiences
        scored = [(self._calculate_informativeness(exp), exp) 
                  for exp in self.trajectories]
        
        # Sort by score (descending) and keep top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        self.trajectories = [exp for _, exp in scored[:self.top_k]]
    
    def retrieve_relevant(self, current_situation: str, k=5) -> List[Dict]:
        """
        Retrieve k most relevant experiences for current situation.
        
        Args:
            current_situation: Current visual observation summary
            k: Number of experiences to retrieve
        
        Returns:
            List of relevant experiences
        """
        if not self.trajectories:
            return []
        
        # Simple relevance: keyword matching
        def relevance_score(experience):
            situation = experience["situation"].lower()
            query = current_situation.lower()
            
            # Count common words
            situation_words = set(situation.split())
            query_words = set(query.split())
            common = len(situation_words & query_words)
            
            # Boost success experiences
            if experience["outcome"] == "success":
                common += 2
            
            return common
        
        # Score and sort
        scored = [(relevance_score(exp), exp) for exp in self.trajectories]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [exp for _, exp in scored[:k]]
    
    def get_all_successes(self) -> List[Dict]:
        """Return all successful trajectories."""
        return [exp for exp in self.trajectories if exp["outcome"] == "success"]
    
    def get_statistics(self) -> Dict:
        """Return memory statistics."""
        return {
            "total_experiences": len(self.trajectories),
            "successes": self.success_count,
            "failures": self.failure_count,
            "unique_failure_patterns": len(self.seen_failures)
        }
    
    def save_to_file(self, filepath: str):
        """Save memory to JSON file."""
        data = {
            "trajectories": self.trajectories,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "seen_failures": list(self.seen_failures)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load memory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.trajectories = data["trajectories"]
        self.success_count = data["success_count"]
        self.failure_count = data["failure_count"]
        self.seen_failures = set(data["seen_failures"])

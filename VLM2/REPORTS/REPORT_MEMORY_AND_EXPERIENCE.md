# VLM2 Agent: Memory Architecture & Experience System

## 1. Overview

The VLM2 Agent uses a **Trajectory-Based Memory** system. Unlike VLM (which uses a Q-Table with state-action values), VLM2 stores **experiential knowledge** in natural language format. The agent learns through experience selection rather than value function approximation.

## 2. Memory Architecture (trajectory_memory.json)

The memory is a list of experience dictionaries, NOT a state-action value table.

### Structure:
```json
{
  "trajectories": [
    {
      "situation": "Goal is down-right, Danger nearby",
      "action": "Action 2",
      "outcome": "success",
      "lesson": "Successfully reached the goal"
    },
    {
      "situation": "Goal is down, Agent visible",
      "action": "Action 1",
      "outcome": "failure",
      "lesson": "Avoid this action in this situation"
    }
  ],
  "success_count": 5,
  "failure_count": 12,
  "seen_failures": ["situation_hash|action_hash", ...]
}
```

### Key Components:

- **situation**: Natural language description of visual observation
  - Derived from VideoPerceptionLayer output
  - Example: "Goal is down-right, Danger nearby"
  
- **action**: The action taken (stored as string for readability)
  - Example: "Action 2" (which is RIGHT in the action space)
  
- **outcome**: Terminal state result
  - Values: `"success"`, `"failure"`, `"timeout"`
  
- **lesson**: What was learned from this experience
  - Successes: "Successfully reached the goal"
  - Failures: "Avoid this action in this situation"

## 3. Experience Selection Mechanism

VLM2 implements **selective memory** based on informativeness:

### Selection Rules:

| Outcome      | Storage Rule                          | Reason                                    |
|--------------|---------------------------------------|-------------------------------------------|
| **Success**  | ALWAYS store                          | Winning strategies are highly valuable    |
| **Failure**  | Store ONLY if novel                   | Avoid redundant failure experiences       |
| **Timeout**  | Discard                               | Non-terminal, not informative             |

### Novelty Detection:

```python
def _is_new_failure(situation, action):
    pattern = f"{situation}|{action}"
    if pattern in seen_failures:
        return False  # Already saw this failure
    seen_failures.add(pattern)
    return True  # Novel failure, store it
```

### Informativeness Scoring:

- Success experiences: **10.0** (highest value)
- Failure experiences: **5.0** (moderate value)
- Other experiences: **1.0** (low value)

When memory exceeds `max_size`, prune to keep top-K most informative experiences.

## 4. Retrieval Mechanism

VLM2 uses **keyword-based relevance matching** to find similar past experiences.

### Retrieval Process:

```python
def retrieve_relevant(current_situation, k=5):
    # 1. Convert current observation to text
    # Example: "Goal is down-right, Danger nearby"
    
    # 2. Calculate relevance for each stored experience
    for experience in memory:
        # Count common words between situations
        common_words = set(experience.situation) & set(current_situation)
        score = len(common_words)
        
        # Boost success experiences
        if experience.outcome == "success":
            score += 2
    
    # 3. Return top-K most relevant
    return sorted_experiences[:k]
```

### Example:

Current situation: `"Goal is down-right, Danger nearby"`

Stored experiences:
1. `"Goal is down-right, Agent visible"` → Relevance: 3 (2 common words + success boost)
2. `"Goal is up, Danger nearby"` → Relevance: 2 (2 common words)
3. `"Agent visible, Movement detected"` → Relevance: 0 (no common words)

Retrieved: Experience #1, #2 (top-2)

## 5. Contrast with VLM Q-Table Approach

| Aspect               | VLM (Q-Table)                       | VLM2 (Trajectory Memory)              |
|----------------------|-------------------------------------|---------------------------------------|
| **Storage Unit**     | State-Action Value (Q(s,a))         | Situational Experience                |
| **Data Format**      | Numerical (float values)            | Natural Language (text)               |
| **Update Rule**      | Bellman Equation (TD learning)      | Selection (keep/discard)              |
| **Memory Growth**    | Fixed size (all state-action pairs) | Dynamic (grows with unique experiences) |
| **Retrieval**        | Direct lookup by state hash         | Similarity matching by keywords       |

### VLM Memory Example:
```json
{
  "hash_of_frame_at_0_0": {
    "RIGHT": 0.8,
    "DOWN": -1.0,
    "LEFT": 0.0,
    "UP": 0.0
  }
}
```

### VLM2 Memory Example:
```json
{
  "situation": "At start position, Goal visible",
  "action": "Action 2",
  "outcome": "success",
  "lesson": "Move RIGHT when at start"
}
```

## 6. Experience Flow Diagram

```
┌─────────────────┐
│  Episode Runs   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Terminal State Reached?     │
│ • Agent on green tile       │
│ • Agent on red tile         │
│ • Max steps reached         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Outcome Inference           │
│ → success / failure         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Selection Rule              │
│ • Success? → STORE          │
│ • Failure & Novel? → STORE  │
│ • Failure & Seen? → DISCARD │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Add to Trajectory Memory    │
│ {situation, action, outcome,│
│  lesson}                    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Memory Pruning              │
│ If size > max_size:         │
│   Keep top-K informative    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Persist to JSON             │
│ trajectory_memory.json      │
└─────────────────────────────┘
```

## 7. Future Enhancements

1. **Visual Embeddings**: Store frame embeddings instead of text summaries
2. **Sequence Patterns**: Recognize multi-step successful trajectories
3. **Similarity Models**: Use semantic similarity (e.g., sentence embeddings) instead of keyword matching
4. **Hierarchical Memory**: Cluster experiences by situation types
5. **VLM Integration**: Feed memories as context to Vision-Language Model for reasoning

## 8. Key Takeaway

VLM2's memory is **experiential and selective**, not **value-based and comprehensive**. It stores "what worked" and "what didn't work" in natural language, making it human-readable and extensible for future VLM integration.

# Video-Based FrozenLake Learning Environment

A complete implementation of a FrozenLake environment where agents learn **ONLY from video input**, without access to symbolic state, grid positions, or direct rewards.

## Architecture

The system follows strict modular separation across 5 component folders:

### 📁 World/ - Physics & Rendering
- **`frozenlake_game.py`** - Game engine (hidden physics)
- **`video_renderer.py`** - RGB frame rendering (Blue=agent, Red=hole, Green=goal)
- **`video_builder.py`** - MP4 video creation from frames

### 📁 Wrapper/ - Interface & Perception
- **`video_environment.py`** - Environment orchestration (agent-facing API)
- **`video_perception.py`** - Computer vision inference (goal direction, danger detection)

### 📁 Verifier/ - Inference Modules
- **`action_inference.py`** - Movement detection from frame comparison
- **`outcome_inference.py`** - Success/failure detection from visual cues

### 📁 Memory/ - Experience Storage
- **`trajectory_memory.py`** - Trajectory-based experience storage
- **`trajectory_memory.json`** - Persistent memory data

### 📁 Client/ - Agents & Applications
- **`demo.py`** - Heuristic agent demonstration

## Hard Constraints (ENFORCED)

❌ **Forbidden:**
- Symbolic state (x,y coordinates, grid arrays, tile IDs)
- Direct reward from environment
- Access to environment internals
- Gym-style state/reward/done

✅ **Required:**
- All learning signals from video frames
- Agent input = images only
- Video-to-structure interpretation

## Running the Demo

```bash
python Client/demo.py
```

This runs a simple heuristic agent for 10 episodes, demonstrating:
- Video-only observation
- Memory-based learning
- Episode video generation
- Experience accumulation

## Output

- **Videos**: `videos/episode_XXXX.mp4` - Visual recording of each episode
- **Memory**: `trajectory_memory.json` - Learned experiences
- **Statistics**: Success rate, memory stats, learned lessons

## Agent Interface

Agents receive:
```python
def agent_function(frame: Image, observation: dict, memories: list) -> int:
    """
    Args:
        frame: Current frame image (PIL Image)
        observation: Perception output (goal_direction, danger_nearby, etc.)
        memories: Relevant past experiences
    
    Returns:
        Action ID (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
    """
    # Agent logic here - NO access to game state!
```

## Design Philosophy

1. **Video as single source of truth**
2. **Simple heuristics over complex models**
3. **Modular, extensible architecture**
4. **No game state leakage**

## File Structure

```
VLM2/
├── World/          - Physics, rendering, video creation
├── Wrapper/        - Environment interface, perception
├── Verifier/       - Action & outcome inference
├── Memory/         - Trajectory memory system
├── Client/         - Demo agent & applications
├── videos/         - Generated episode videos
└── REPORTS/        - Documentation
```

## Dependencies

```bash
pip install pillow opencv-python numpy
```

## Success Criteria

The system is correct if:
- Agent improves over episodes
- Using only video input
- Without accessing environment internals
- Without explicit reward signals

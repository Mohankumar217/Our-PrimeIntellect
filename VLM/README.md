# Prime-Intellect VLM-Style FrozenLake Agent

This repository implements a **VLM-based** variant of the Prime-Intellect FrozenLake agent. 
It strictly adheres to the ideology that **learning happens via selection and memory reuse**, not gradient updates.

## Architecture

The system is composed of the following modules:

### 1. World (`World/frozenlake_world.py` & `frozenlake_renderer.py`)
- **Logic**: Pure symbolic state (grid, position, rules).
- **Renderer**: Converts the symbolic state into a visual RGB image (PIL).
- **Constraint**: No text labels (coordinates, danger hints) are ever shown in the image.

### 2. Wrapper (`Wrapper/vlm_wrapper.py` & `xml_parser.py`)
- **Multimodal Prompting**: Constructs prompts containing System Instructions, Episodic Memories, and the Current Image.
- **Parsing**: Extracts `<action>` tags from the VLM's textual output.
- **Safety**: Ensures no symbolic state leaks into the prompt.

### 3. Evaluation (`Evaluation/outcome.py` & `efficiency.py`)
- Verifies success (Goal reached).
- Penalizes failure (Hole).
- Scores efficiency (Step count).

### 4. Client (`Client/run_agent.py`)
- **Orchestrator**: Runs the agent loop (Observe -> Reason -> Act).
- **Memory Manager**: Loads past successful episodes (`Memory/memory.json`) and injects them into the prompt.
- **Selection**: Only high-scoring episodes are saved to memory.

## Comparison: LLM vs VLM Style

| Aspect | LLM Style | VLM Style |
|qs|---|---|
| **Observation** | Text (Coordinates, Lists) | Image (RGB Pixels) |
| **Memory** | Rules / Paths | Visual Experiences |
| **Learning** | Strategy Reuse | Perception Reuse |
| **Training** | None | None |

## Usage

Run the agent loop:
```bash
python Client/run_agent.py
```

## Ideology
- **No Training**: The model weights are frozen.
- **Selection**: We only keep what survives.
- **Wrapper**: Controls perception (what the model sees) and action (what the model can do).

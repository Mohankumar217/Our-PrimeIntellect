# FrozenLake PI

A custom, robust evaluation (eval) environment designed to test the spatial reasoning and planning capabilities of Large Language Models (LLMs). This project decouples the simulation physics from the agent interface, creating a pure text-based playground for Generative AI.

---

## 1. Rules of the Game
The "Frozen Lake" is a classic grid-world puzzle. The rules are simple but unforgiving:

- **The Grid**: A 4x4 map of tiles.
- **The Objective**: Navigate from the **Start (S)** tile to the **Goal (G)** tile.
- **Hazards**: Some tiles are **Holes (H)**. If you step on a Hole, you fall in and the game is over (Loss).
- **Safe Ground**: **Frozen (F)** tiles are safe to walk on.
- **Movement**: You can move in four directions: `UP`, `DOWN`, `LEFT`, `RIGHT`.
- **Winning**: You win ONLY if you land on the **Goal (G)** tile.

### The Map Symbols
| Symbol | Meaning | Consequence |
| :---: | :--- | :--- |
| **S** | Start | You begin here. Safe. |
| **F** | Frozen | Safe to walk on. |
| **H** | Hole | **Game Over** (You die). |
| **G** | Goal | **Victory** (You win). |

---

## 2. How Our Agents See the Game
Unlike human players who see a 2D visual grid, our LLM agents interact with the world purely through **Text**.

### The Observation (Input to Agent)
The agent does *not* see a matrix. Instead, it receives a detailed natural language description of its current state.

**Example Prompt:**
```text
System: You are navigating a frozen lake.
Observation: You are currently at position (0,0).
- To your RIGHT is a Frozen tile (Safe).
- To your DOWN is a Frozen tile (Safe).
- To your LEFT is a Wall.
- To your UP is a Wall.
History: You have taken 0 steps.
```

### The Action (Output from Agent)
The agent effectively "plays" by generating a structured text response. We use XML tags to parse the intended move.

**Example Response:**
```xml
Thought: I need to move towards the center. The right path looks clear.
<action>RIGHT</action>
```

---

## 3. What This Environment Does
This codebase (`frozenlake_pi`) is a bespoke implementation of the environment, built from scratch to avoid dependencies on `Gymnasium` or `OpenAI Gym`. It is specifically engineered for **LLM Evaluation**.

### Key Features
*   **Text Wrapper Layer**: A translation layer that converts raw `(x, y)` coordinates into rich semantic descriptions ("You hit a wall").
*   **Deterministic Physics**: The environment is non-slippery by default (unlike the standard Gym version). If you press "RIGHT", you *will* go Right. This isolates the agent's **Reasoning Logic** from randomness.
*   **In-Context Learning Support**: The training loop supports "Few-Shot" learning. We can feed successful past episodes back into the prompt to see if the agent can "learn" from its own history.
*   **Strict Verification**: A dedicated Scoring Rubric (`verifier/`) checks not just if the agent won, but how efficient its path was and whether it followed the strict output format instructions.

### Why did we build this?
Standard RL environments output numerical vectors (e.g., `[0, 0, 1, 0]`). LLMs cannot read vectors intuitively. This environment bridges the gap, allowing us to benchmark models like **Gemini**, **GPT-4**, and **Qwen** on their ability to maintain a "mental map" and plan multi-step paths using only text.

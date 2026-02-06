VLM Agent: Reward Mechanism & Memory Architecture

1. Overview
The VLM Agent (Prime-Intellect Architecture) uses a Visual Q-Table memory system. Instead of training a neural network policy (Deep RL), we maintain an explicit Tabular Value Function that guides a frozen VLM. The VLM acts as the "Policy Network" by interpreting these values in context.

2. Memory Architecture (memory.json)
The memory is a Dictionary (JSON) mapping States to Action Values.

Structure:
{
  "(row, col)": {
    "UP": float,    // Q-Value for moving UP
    "DOWN": float,  // Q-Value for moving DOWN
    "LEFT": float,  // Q-Value for moving LEFT
    "RIGHT": float  // Q-Value for moving RIGHT
  }
}

- State Key: Currently represented by the (x, y) coordinate tuple. In future visual-only iterations, this will be replaced by a perceptual hash of the image.
- Action Value: A scalar score representing the expected future reward.

3. Reward Mechanism
We invoke a custom reward function that balances Goal Acquisition with Efficiency (Step Cost).

Reward Function R(s, a, s'):
Condition | Reward Value
- Goal Reached: +1.0
- Fell in Hole: -1.0
- Moved Closer (Manhattan Dist Decrease): +0.1 (Shaping)
- Moved Away (Manhattan Dist Increase): -0.1 (Shaping)
- Hit Wall / Neutral: -0.1 (Step Cost)

Note: The proximity shaping (+0.1/-0.1) helps the agent learn directional gradients even before reaching the goal.

4. Q-Learning Update Rule
We apply the standard Q-Learning (Off-Policy TD Control) update rule after every step.

Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [ R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t) ]

Hyperparameters:
- alpha (Learning Rate): 0.1 (Slow, stable updates)
- gamma (Discount Factor): 0.9 (High foresight)

5. The "In-Context" Policy
The Agent does not mechanically select argmax Q(s,a).
Instead, the values are injected into the Prompt:
"Memory suggests: RIGHT (0.8), DOWN (-1.0)."

The VLM (Model) then performs the final decision, allowing it to:
1. Exploit: Follow the high Q-value (RIGHT).
2. Explore: Try a neutral action if it "sees" a potential shortcut.
3. Override: Ignore a false positive Q-value if the visual evidence (e.g., a hole) contradicts it.

🔄 **The main control engine (Environment.py)**

At the heart of the framework is the Episodic class, which manages the step-by-step operational lifecycle of any connected agent. Highly modular by design, this class enforces a strict, cyclical pipeline during each step:

_Observe:_ Fetch real-time or cached environmental data using abstract sensor objects equipped on the agent, and execute any vision models to further process data for downstream tasks.

_Think (Optional VLA):_ Route complex visual/textual data through Vision-Language-Action models for high-level reasoning and intent generation.

_Decide (DRL):_ Determine the optimal next action based on the current policy.

_Act:_ Translate the chosen action into physical execution using a low-level PID motor controller.

_Evaluate:_ Apply DRL reward functions and evaluate termination conditions to conclude or continue the episode.

🤖 **The Abstract Agent Interface (Agent.py)**

The abstract Agent class serves as the ultimate bridge between the episodic environment and the physical/simulated world. It defines all interactions through linked abstract action and sensor objects.

Currently, the framework supports two primary backend environments out of the box:

Microsoft AirSim: For high-fidelity, real-time simulated physics and vision testing.

Discrete Data-Map Cache: A high-speed execution mode that uses pre-cached observations at all (x, y, z) coordinates. This allows for lightning-fast DRL inference times and safely obfuscates low-level hardware control during heavy training loop

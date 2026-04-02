🔄 **The main control engine (Environment.py)**

At the heart of the framework is the episodic Environment class, which manages the step-by-step operational lifecycle of any connected agent. Highly modular by design, this class enforces a strict, cyclical pipeline during each step:

_Observe:_ Fetch real-time or cached environmental data using abstract sensor objects equipped on the agent, and execute any vision models to further process data for downstream tasks.

_Think (Optional VLA):_ Route complex visual/textual data through Vision-Language-Action models for high-level reasoning, by translating progress of the agent as it progress through the map along with a visual representation of its vicinity and progress (though world building and ray tracing).

_Decide (DRL):_ Determine the optimal next action based on the current policy.

_Act:_ Translate the chosen action into physical execution using a low-level PID motor controller.

_Evaluate:_ Apply DRL reward functions and evaluate termination conditions to conclude or continue the episode.

🤖 **The Abstract Agent Interface (Agent.py)**

The abstract Agent class serves as the ultimate bridge between the episodic environment and the physical/simulated world. It defines all interactions through linked abstract action and sensor objects.

Currently, the framework supports two primary backend environments out of the box:

_MicrosoftAirSim_ For high-fidelity, real-time simulated physics and vision testing.

_DataMap_ A high-speed execution mode that uses pre-cached observations at all (x, y, z) coordinates. This allows for lightning-fast DRL inference times and safely obfuscates low-level hardware control during an otherwise heavy training and testing loop.

🛠️ **Setup**

1. Set your python environment variable to point to the home directory which this repository lives, so that it can import modules from OmniNaviPy.

2. Download the required data to run the _MicrosoftAirSim_ agent, as found on Microsoft's release page here: https://github.com/microsoft/airsim/releases, unzip and note the path to either the .sh or .exe executable file for later. You will also need to download Microsoft Visual Studio Code with C++ development, and directx (if on Windows).

3. Downlad the required data to run the _DataMap_ agent. This can be found on the public Dropbox at: https://www.dropbox.com/scl/fo/9zibc89vm58ypqfdl7vgz/AOuoLZtU-VuOPdJoEVXsFIc?rlkey=hc74tvl9ui045kwd7fomien35&st=r64plkhb&dl=0. Specifically you need the AirSimNH data and the DQN_beta policy to run **Firt Run** below. Drop the sub-folders directly into the github parent folder such as: OmniNaviPy/data and OmniNaviPy/policies. Note that if you follow the samea data structure, that this agent works with any collection of data for any environment (real or sim).

4. Each agent has a different python virtual environment, conda. Particularly pesky is the _MicrosoftAirSim_ agent which has compadibliity issues with outdated and depreciated tornado, msgpackrpc, and airsim third-party python libraries. You can find these in the envs subfolder of this repository, which are given as both requirements.txt and environment.yml files to install on your local computer via conda or some other virtual python environment. Note that the environment may require a specific pip version along withy python version, see the top of the requirements.txt file for both of these. Macke sure to activate the proper environment first before running any python scripts that use that specific agent!

🚀 **First Run**

Open the evaluate_navigation.py file and set the path to your unzipped AirSim executable as noted from before, on line 6: release_path='path/to/your/.exe/or/.sh'

Run the evaluate_navigation.py file to test run everything, which by default will use the _DataMap_ agent and a two-tier navigational framework that uses the DQN_beta policy (previosly trained via DRL) to direct the drone which actions to take. Feel free to play with the paramters after that. Happy navigating!

Note that this uses a static hold out set of trajectories to evaluate on, which should be held out from training all future policies. This creates a consistent testing metric across different configurations for robust model evaluations.

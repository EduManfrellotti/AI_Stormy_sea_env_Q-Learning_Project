from sea_env import CustomGridWorldEnv
import gymnasium as gym

"""
This script demonstrates how to interact with the custom Gym environment implemented.
In particular:

1. Initializes the environment with human rendering.
2. Runs a simulation for some random sampled steps to give an example.
3. Prints information about the environment and resets it upon termination or truncation.

The script helps in testing the environment's behavior and check for errors.
"""

# Environment initialization
render_mode = 'human'
env = gym.make('CustomGridWorld-v0', render_mode=render_mode)

# Set to 1 the fps for better rendering
env.metadata["render_fps"] = 1

# Reset the environment and get the starting observation and info
obs, info = env.reset()
print("Initial Observation:", obs, "\nInfo:", info, "\n")

steps = 50   # Number of steps to run the simulation

# Testing loop
for step in range(steps):
    # Take a random sampled action and return the output
    obs, reward, _, _, info = env.step(env.action_space.sample())

    # Print the step information
    print("Info:", info)

    # Verify the behaviour of truncation / termination flags
    if info["Termination"] or info["Truncation"]:
        obs, info = env.reset()
        print("Info:", info, "\n")

# Close the environment
env.close()

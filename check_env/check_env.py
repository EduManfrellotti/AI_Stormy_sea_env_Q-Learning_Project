from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from sea_env import CustomGridWorldEnv

"""
This script is designed to verify the validity of a custom Gym environment to eventually train an agent on it using stable_baseline3.

USAGE:

1. Instantiate the custom environment.
2. Use the `check_env` function to perform checks.
3. Review any warnings or errors output by the function to ensure the environment is correctly implemented.
"""

# Initialize the custom environment and checks it
env = CustomGridWorldEnv()
check_env(env)

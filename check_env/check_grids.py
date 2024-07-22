from sea_env import CustomGridWorldEnv
import gymnasium as gym
from vortices import Vortex
import numpy as np

"""
This script is designed to:

1. Create and inspect the custom Gym environment implemented.
2. Create and examine vortex initialized grids to check for errors.
"""

# Create, reset and test the initialized grids of the environment to check for errors
env = gym.make('CustomGridWorld-v0', render_mode="human")
env.reset()
print(env.prob_grid)
print("\n", env.dir_grid, "\n")

# Test the initialization of both types of vortices
v = Vortex(4, "clockwise")
print(v.v_prob)
print("\n", v.v_dir)
print("\n", v.v_colors, "\n")

v1 = Vortex(4, "cclockwise")
print(v1.v_prob)
print("\n", v1.v_dir)
print("\n", v1.v_colors)

import os
import gymnasium as gym
import numpy as np
import random
import pickle
from sea_env import CustomGridWorldEnv
from make_plots import *


def run_ql(env, config, episodes, is_training=True):
    """
    Runs the Q-Learning algorithm to train or test the agent on a configuration of the environment.

    PARAMETERS:

    - env (gym.Env): The environment to run the Q-Learning algorithm on.
    - config (str): The configuration name for the current training/test run.
    - episodes (int): The number of episodes to run.
    - is_training (bool): Flag indicating whether the run is for training or testing.

    If is_training is True, the function will train a Q-Learning model and save it along with statistics.
    If is_training is False, the function will load a pre-trained model and run a test.

    The function saves the Q-Learning model and statistics to specified directories.
    """

    # Create directories to save models, plots, and statistics
    model_dir = f"models/QL/{config}"
    plots_dir = f"plots/QL/{config}"
    stats_dir = f"stats/QL/{config}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Load the Q-Learning model if testing
    if not is_training:
        with open(f'{model_dir}/model_final.pkl', 'rb') as f:
            q = pickle.load(f)

    # If training initialize the Q-table
    else:
        q = np.zeros((env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.action_space.n))

    # Set Q-Learning parameters
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1

    # Dictionary to collect statistics
    stats = {
        "mean_ep_len": [],      # Mean episode length
        "mean_ep_rew": [],      # Mean episode reward
        "mean_term": [],        # Mean episode number of terminations
        "mean_trunc": []        # Mean episode number of truncations
    }

    # Temporary value to store single episode statistics
    steps = []
    rewards = []
    terminations = 0
    truncations = 0

    # Training loop
    for i in range(episodes):
        print(f'Episode {i}')

        # Reset the environment and get the initial state
        state = env.reset()[0]

        while True:
            # Exploration: if training, perform an epsilon-greedy action selection
            if is_training and random.random() < epsilon:
                action = env.action_space.sample()

            # Exploitation: if testing, select the action with the highest Q-value for the current state
            else:
                q_state_idx = tuple(state["agent"]) + tuple(state["goal"])
                action = np.argmax(q[q_state_idx])

            # Perform the selected action and get the resulting state and reward
            new_state, reward, _, _, info = env.step(action)
            q_state_action_idx = tuple(state["agent"]) + tuple(state["goal"]) + (action,)
            q_new_state_idx = tuple(new_state["agent"]) + tuple(new_state["goal"])

            # Update the Q-table if in training mode
            if is_training:
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx]
                )

            # Update current state
            state = new_state

            print("info:", info)

            # Check if the episode is terminated or truncated
            if (info["Termination"] or info["Truncation"]) == 1:
                steps.append(info["Num_moves"])
                rewards.append(info["Ep_reward"])
                terminations += int(info["Termination"])
                truncations += int(info["Truncation"])
                print("")
                break

        # Gradually reduce epsilon to shift from exploration to exploitation
        epsilon = max(epsilon - 1 / episodes, 0)

        # Every 500 episodes, save statistics and the model
        if is_training and (i + 1) % 500 == 0:
            # Calculate and store average statistics for the last 500 episodes
            stats["mean_ep_len"].append(np.mean(steps))
            stats["mean_ep_rew"].append(np.mean(rewards))
            stats["mean_term"].append(terminations / 500)
            stats["mean_trunc"].append(truncations / 500)

            # Reset lists and counters for the next 500 episodes
            steps = []
            rewards = []
            terminations = 0
            truncations = 0

            # Save the Q-table model
            with open(os.path.join(model_dir, f'model_{i + 1}.pkl'), 'wb') as f:
                pickle.dump(q, f)

            # Save the statistics
            with open(os.path.join(stats_dir, f'stats_{i + 1}.pkl'), 'wb') as f:
                pickle.dump(stats, f)

    # Save the final model and statistics after all episodes
    if is_training:
        with open(os.path.join(model_dir, f'model_final.pkl'), 'wb') as f:
            pickle.dump(q, f)

        with open(os.path.join(stats_dir, f'stats_final.pkl'), 'wb') as f:
            pickle.dump(stats, f)

    # Close the environment
    env.close()


def load_stats(config, episodes):
    """
    Loads and returns statistics from a file.

    PARAMETERS:

    - config (str): The configuration name for the current training/testing run.
    - episodes (int): The number of episodes.
    """
    # Select the file to load
    stats_dir = f"stats/QL/{config}"
    stats_file = os.path.join(stats_dir, f'stats_{episodes}.pkl')

    # Load the statistics from file
    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)

    return stats


def init_env(config, r_mode):
    """
    Initializes and returns the environment based on the provided configuration.

    PARAMETERS:

    - config (str): The configuration name for the environment.
    - r_mode (str): The rendering mode for the environment.

    The possible types of environments are:
    -1 Easy environment: basic configuration, agent goal and vortices
    -2 Easy randomized environment: basic configuration with randomized goal's and agent's positions
    -3 Hard environment: basic configuration with rocks as obstacles
    -4 Randomized hard environment: hard configuration with some rocks randomized according to two different patterns
    -5 Hard-randomized hard environment: randomized hard configuration with randomized goal's and agent's positions
    """

    # Create the environment based on the specified configuration
    if config == "easy_env":
        env = gym.make('CustomGridWorld-v0', render_mode=r_mode)
    elif config == "r_easy_env":
        env = gym.make('CustomGridWorld-v0', render_mode=r_mode, rand_a=True, rand_g=True)
    elif config == "hard_env":
        env = gym.make('CustomGridWorld-v0', render_mode=r_mode, add_rocks=True)
    elif config == "r_hard_env":
        env = gym.make('CustomGridWorld-v0', render_mode=r_mode, add_rocks=True, rand_r=True)
    elif config == "rr_hard_env":
        env = gym.make('CustomGridWorld-v0', render_mode=r_mode, rand_a=True, rand_g=True, add_rocks=True, rand_r=True)

    # Raise an error if the configuration is not recognized
    else:
        raise ValueError(f"The configuration '{config}' has not been implemented yet")

    return env


def train_ql(config, episodes):
    """
    Trains the agent on the provided environment's configuration using Q-Learning for a number of input episodes

    PARAMETERS:

    - config (str): The configuration name for the environment.
    - episodes (int): The number of episodes to train.
    """

    env = init_env(config, None)   # Initialize the environment without rendering (for faster training)
    run_ql(env, config=config, episodes=episodes, is_training=True)  # Run Q-Learning training


def test_ql(config, episodes):
    """
    Tests the agent on the provided environment's configuration using Q-Learning for a number of input episodes.

    PARAMETERS:

    - config (str): The configuration name for the environment.
    - episodes (int): The number of episodes to test.
    """

    env = init_env(config, "human")   # Initialize the environment with human rendering
    env.metadata["render_fps"] = 4   # Set a lower number of frames for better rendering
    run_ql(env, config=config, episodes=episodes, is_training=False)   # Run Q-Learning testing


if __name__ == '__main__':
    """
    Main function to train and test the Q-Learning algorithm on various configurations.
    """

    # Define the list of possible configurations, the number of training episodes and the number of testing episodes
    configs = ["easy_env", "r_easy_env", "hard_env", "r_hard_env", "rr_hard_env"]
    train_ep = 20000
    test_ep = 1

    # Uncomment to train, plot statistics, and test for each configuration

    # Easy enviroment
    # train_ql(configs[0], train_ep)
    # stats = load_stats(configs[0], train_ep)
    # plot_stats(configs[0], "QL", stats, train_ep)
    # test_ql(configs[0], test_ep)

    # Easy enviroment with random goal and agent
    # train_ql(configs[1], train_ep)
    # stats = load_stats(configs[1], train_ep)
    # plot_stats(configs[1], "QL", stats, train_ep)
    test_ql(configs[1], test_ep)

    # Hard enviroment with rocks
    # train_ql(configs[2], train_ep)
    # stats = load_stats(configs[2], train_ep)
    # plot_stats(configs[2], "QL", stats, train_ep)
    # test_ql(configs[2], test_ep)

    # Hard enviroment with random rocks
    # train_ql(configs[3], train_ep)
    # stats = load_stats(configs[3], train_ep)
    # plot_stats(configs[3], "QL", stats, train_ep)
    # test_ql(configs[3], test_ep)

    # Hard enviroment with random rocks, agent and goal
    # train_ql(configs[4], train_ep)
    # stats = load_stats(configs[4], train_ep)
    # plot_stats(configs[4], "QL", stats, train_ep)
    # test_ql(configs[4], test_ep)

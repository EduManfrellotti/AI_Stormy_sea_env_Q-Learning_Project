import os
import matplotlib.pyplot as plt


def plot_stats(config, algorithm, stats, num_ep):
    """
    Generate and save plots for various statistics collected over episodes during training.

    PARAMETERS:

    - config (str): The configuration name for the current training run.
    - algorithm (str): The name of the algorithm used for training.
    - stats (dict): A dictionary containing statistical data for plotting.
        - "mean_ep_len": List of average episode lengths.
        - "mean_ep_rew": List of average episode rewards.
        - "mean_term": List of average terminations.
        - "mean_trunc": List of average truncations.
    - num_ep (int): The total number of episodes.

    The function creates plots for:
    - Average episode length
    - Average episode reward
    - Average number of terminations and truncations
    """

    # Define the directory where the plots will be saved
    plots_dir = f"plots/{algorithm}/{config}"
    os.makedirs(plots_dir, exist_ok=True)

    # Create a list of episode numbers for the x-axis
    episodes = [i for i in range(500, num_ep + 1, 500)]

    # Font sizes for the plot titles, labels, and ticks
    fs_title = 20
    fs_label = 15
    fs_ticks = 13

    # Plot for Average episode length
    plt.figure(figsize=(15, 8))
    plt.plot(episodes, stats["mean_ep_len"], color='b', linestyle='-', marker='o')
    plt.title('Average episode Length', fontsize=fs_title)
    plt.xlabel('Episode number', fontsize=fs_label)
    plt.ylabel('Episode Length', fontsize=fs_label)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'avg_ep_len_{num_ep}.png'))
    plt.show()
    plt.close()

    # Plot for Average episode reward
    plt.figure(figsize=(15, 8))
    plt.plot(episodes, stats["mean_ep_rew"], color='r', linestyle='-', marker='o')
    plt.title('Average final reward over episodes', fontsize=fs_title)
    plt.xlabel('Episodes', fontsize=fs_label)
    plt.ylabel('Cumulative Reward', fontsize=fs_label)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'avg_ep_rew_{num_ep}.png'))
    plt.show()
    plt.close()

    # Plot for Average number of terminations and truncations
    plt.figure(figsize=(15, 8))
    plt.plot(episodes, stats["mean_term"],  label='Termination', color='g', linestyle='-', marker='o')
    plt.plot(episodes, stats["mean_trunc"],  label='Truncation', color='r', linestyle='-', marker='x')
    plt.title('Average number of victories and failures over episodes', fontsize=fs_title)
    plt.xlabel('Episode number', fontsize=fs_label)
    plt.ylabel('Terminations and truncations', fontsize=fs_label)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'avg_term_trunc_{num_ep}.png'))
    plt.show()
    plt.close()

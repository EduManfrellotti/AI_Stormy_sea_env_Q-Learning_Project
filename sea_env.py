import random
import numpy as np
import pygame                                       # For rendering
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register    # To register the enviroment
from vortices import Vortex                         # To initialize the vortices

# Register the custom environment with Gym.
# This registration is essential for Gym to easily find and instantiate the environment using a unique identifier.
register(
    id="CustomGridWorld-v0",
    entry_point="__main__:CustomGridWorldEnv",
    max_episode_steps=450,
)


class CustomGridWorldEnv(gym.Env):
    """
    CustomGridWorldEnv is a custom Gym environment designed to simulate a sea grid world scenario.
    This environment features a grid-based map with currents that can lead into two strategically placed vortices.

    Based on the input configuration the environment can contain rocks or introduce randomness of some of the elements.

    The objective of the agent is to reach the goal without exceeding the maximum number of steps or being trapped inside
    one of the vortices.

    This setup encourages the agent to develop the optimal strategy to successfully solve the proposed scenario.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": float('inf')}

    def __init__(self, render_mode=None, grid_size=10, add_rocks=False, rand_a=False, rand_g=False, rand_r=False):
        """
        Initializes the environment.

        PARAMETERS:

        - render_mode (str, optional): Specifies the mode for rendering the environment.
        - grid_size (int): Defines the size of the square grid used for the environment.
        - add_rocks (bool): If True, the environment will include rocks as obstacles.
        - rand_a (bool): If True, the starting position of the agent will be randomized.
        - rand_g (bool): If True, the goal's position will be randomized.
        - rand_r (bool): If True, the positions of the rocks will be randomized according to predefined patterns.
        """

        # VISUAL SETTINGS
        self.grid_size = grid_size    # Size of the environment grid
        self.window_size = 512        # Size of the rendering window

        self.window = None            # Pygame window for rendering
        self.clock = None             # Pygame clock for controlling the frame rate

        # Set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # ENVIROMENT SETTINGS
        self.max_episode_steps = 450     # Maximum number of steps before the truncation of the episode

        # Set the randomness of the environment
        self.add_rocks = add_rocks
        self.random_a = rand_a
        self.random_g = rand_g
        self.random_rocks = rand_r

        # Initialize vortices
        self.v_size = 4     # Vortex size
        self.vortices = np.array([Vortex(self.v_size, "clockwise"),      # Number of vortices and their type
                                  Vortex(self.v_size, "cclockwise")])
        self.start = np.array([1, 5])       # Starting cells for the first element of the vortices grids

        # Initialize matrices for directions, colors, probabilities
        self.prob_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.dir_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.colors_grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # OBSERVATION SPACE
        # Define the observation space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),  # position of the agent
            "goal": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),   # position of the goal
        })

        # ACTIONS
        # Define the action space (4 possible movements)
        self.action_space = spaces.Discrete(4)

        # REWARDS
        # Define the rewards
        self.rewards_d = {
            "goal": 10000,          # Highest reward for reaching the goal
            "illegal": -10,         # Small penalty for an illegal move
            "vortex": -100,         # Medium penalty for being trapped inside a vortex
            "max_steps": -500,      # Highest penalty to exceed the maximum number of steps
            "no_move": -50,         # Precautional penalty to ensure that the agent keep moving
        }

        # COLORS FOR RENDERING
        # Define colors in an RGB format to be used to render the environment
        self.colors = {
            0: (53, 145, 214),      # Open sea (Blue)
            1: (168, 176, 178),     # Agent ship (Gray)
            2: (255, 203, 62),      # Goal lighthouse (Yellow)
            3: (75, 54, 32),        # Rocks (Brown)
        }

    # ----------------------------------------- INFORMATION METHODS ---------------------------------------------------------

    def get_obs(self):
        """
        Returns the observation as a dictionary containing the position of the agent and the position of the goal
        """
        observation = {
            "agent": self.agent_loc,
            "goal": self.goal_loc,
        }

        return observation

    def get_info(self):
        """
        Returns a dictionary containing additional information to monitor the enviroment's state during the training
        """
        info = {
            "Distance_to_goal": np.linalg.norm(self.agent_loc - self.goal_loc, ord=1),    # Distance of the agent from the goal
            "Num_moves": self.num_moves,       # Current number of moves performed
            "Step_reward": self.reward,        # Reward of the current step
            "Ep_reward": self.ep_reward,       # Current cumulative reward of the episode
            "Termination": self.termination,   # Boolean to state if the agent reached the goal
            "Truncation": self.truncation,     # Boolean to state if the episode has been truncated
        }

        return info

    # --------------------------------------------- ENVIROMENT METHODS --------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the first observation and info of the episode
        """
        super().reset(seed=seed, options=None)    # Gives the possibility to set a seed, for reproducibility of the results

        # Reset the rocks if necessary
        if self.add_rocks:
            self.rocks_pos = self.init_rocks()

        # Reset the agent's position
        self.agent_loc = self.init_agent()

        # Reset the goal's position
        self.goal_loc = self.init_goal()

        # Reset the direction grid
        self.init_currents()

        # Reset the probability grid
        self.init_probs()

        # Reset reward, cumulative episode reward, termination and truncation flags
        self.reward = 0
        self.ep_reward = 0
        self.termination = False
        self.truncation = False

        # Reset move counter
        self.num_moves = 0

        # Render the current state of the enviroment
        self.update_render()

        # Reset info and observations dictionaries
        self.observation = self.get_obs()
        self.info = self.get_info()

        # Render the enviroment in its initial state
        if self.render_mode == "human":
            self.render_frame()

        return self.observation, self.info

    def step(self, action):
        """
        Execute a step in the environment based on the chosen action and assign the corresponding reward.

        The action performed by the agent can either be:
        - The intended action chosen by the agent.
        - The direction influenced by the current specified in the agent's current position cell of the direction grid.

        The action is chosen by random sampling a probability value and looking at the agent's position cell in the
        probability grid. If this value is lower than the probability in the considered cell then the agent will follow the
        direction of the current, otherwise it will make its chosen action.
        """
        # Reset step reward
        self.reward = 0

        prev_loc = self.agent_loc.copy()       # Copy of the current position of the agent
        new_loc = self.move_agent(action)      # New position based on agent's chosen action
        cur_prob = self.prob_grid[self.agent_loc[0], self.agent_loc[1]]  # Probability for the agent to follow the current
        current_loc = self.apply_current()     # New position based on the cell of the direction grid

        # MOVE THE AGENT
        # Sort the probability to be trapped by the current
        if random.random() <= cur_prob and self.check_move(current_loc):
            # Agent follows vortex direction
            self.agent_loc = current_loc
        else:
            # Apply the intended action
            if self.check_move(new_loc):     # Check for illegal action
                self.agent_loc = new_loc
            else:
                self.reward += self.rewards_d["illegal"]

        # Check if the agent didn't move
        if np.array_equal(self.agent_loc, prev_loc):
            self.reward += self.rewards_d["no_move"]

        # Update number of moves
        self.num_moves += 1

        # Save the current probability of being trapped by the current
        new_prob = self.prob_grid[self.agent_loc[0], self.agent_loc[1]]

        # TERMINATION / TRUNCATION CONDITIONS
        # Check if agent has reached the goal
        if np.array_equal(self.agent_loc, self.goal_loc):
            self.reward += self.rewards_d["goal"]
            self.termination = True

        # Check if the agent is arrived in the center of the vortex
        if new_prob == 0:
            self.reward += self.rewards_d["vortex"]
            self.truncation = True

        # Check for truncation based of the current number of moves
        if self.num_moves >= self.max_episode_steps and not self.termination:
            self.reward += self.rewards_d["max_steps"]
            self.truncation = True

        # Update episode reward
        self.ep_reward += self.reward

        # Get the observation and the info for the current step
        observation = self.get_obs()
        info = self.get_info()

        # Update the rendering
        if self.render_mode == "human":
            self.update_render()
            self.render_frame()

        return observation, self.reward, self.termination, self.truncation, info

    def init_rocks(self):
        """
        Assign rocks position in the grid, based on the chosen configuration of the environment.

        Fixed rocks are found in the opposite quadrants of the vortices, in the upper right and lower left grid's corners.
        The remaining four rocks are placed following two fixed patterns based on the value of the random rocks flag.
        If True, one of the two patterns is randomly chosen, otherwise the rocks are placed following the first pattern.
        """
        # FIXED ROCKS
        left_c = np.array([self.grid_size - 1, 0])      # Lower left corner
        right_c = np.array([0, self.grid_size - 1])     # Upper right corner

        # Define the array containing all rocks positions
        rocks_pos = np.array([
            left_c,
            right_c,
            left_c + np.array([-1, 0]),
            left_c + np.array([0, 1]),
            right_c + np.array([1, 0]),
            right_c + np.array([0, -1]),
            [6, 2],
            [7, 3],
            [2, 7],
            [3, 6],
        ])

        # RANDOM ROCKS
        if self.random_rocks:
            choice = random.randint(1, 2)

            # Pattern 1
            if choice == 1:
                additional_rocks = np.array([[0, 3], [9, 6]])
            # Pattern 2
            else:
                additional_rocks = np.array([[3, 0], [6, 9]])

            rocks_pos = np.append(rocks_pos, additional_rocks, axis=0)
        else:
            # Pattern 1
            fixed_rocks = np.array([[0, 3], [9, 6]])
            rocks_pos = np.append(rocks_pos, fixed_rocks, axis=0)

        return rocks_pos

    def init_agent(self):
        """
        Returns the initial position of the agent based on the chosen configuration of the environment.

        Flags for random agent and random rocks are checked. The possible combinations are:
        -1. Fixed agent: the agent is placed in [0, 0]
        -2. Random agent - no rocks: the agent is placed in a random position on the first column
        -3. Random agent - rocks: the agent is placed in a random position on the first column avoiding rocks positions.
        """
        # RANDOM AGENT
        if self.random_a:
            # Configuration 3
            if self.add_rocks:
                while True:
                    random_x = random.randint(0, self.grid_size - 1)   # Sample the agent row
                    agent_loc = np.array([random_x, 0])

                    # Break if a valid position is found
                    if not any(np.array_equal(agent_loc, rock_pos) for rock_pos in self.rocks_pos):
                        break
            # Configuration 2
            else:
                random_x = random.randint(0, self.grid_size - 1)
                agent_loc = np.array([random_x, 0])

        # FIXED AGENT
        # Configuration 1
        else:
            agent_loc = np.array([0, 0])

        return agent_loc

    def init_goal(self):
        """
        Returns the initial position of the goal based on the chosen configuration of the environment.

        Flags for random goal and random rocks are checked. The possible combinations are:
        -1. Fixed goal: the agent is placed in [9, 9]
        -2. Random goal - no rocks: the goal is placed in a random position on the last column
        -3. Random goal - rocks: the goal is placed in a random position on the last column avoiding rocks positions.
        """
        # RANDOM GOAL
        if self.random_g:
            # Configuration 3
            if self.add_rocks:
                while True:
                    random_x = random.randint(0, self.grid_size - 1)    # Sample the goal row
                    goal_loc = np.array([random_x, self.grid_size - 1])

                    # Break if a valid position is found
                    if not any(np.array_equal(goal_loc, rock_pos) for rock_pos in self.rocks_pos):
                        break
            # Configuration 2
            else:
                random_x = random.randint(0, self.grid_size - 1)
                goal_loc = np.array([random_x, self.grid_size - 1])

        # FIXED GOAL
        else:
            goal_loc = np.array([self.grid_size - 1, self.grid_size - 1])

        return goal_loc

    def check_move(self, target):
        """
        Returns a boolean value based on the target position of the agent's step.

        This functions returns False if the agent try to move outside the grid or to go in a position occupied
        by a rock (if present), otherwise it returns True.
        """
        # Check if target is within the grid
        if any(target < 0) or any(target >= self.grid_size):
            return False

        if self.add_rocks:
            # Check if target is not in rocks positions
            for rock_pos in self.rocks_pos:
                if np.array_equal(target, rock_pos):
                    return False

        return True

    # ---------------------------------------------- WATER INTERACTION METHODS -----------------------------------------------

    def move_agent(self, action):
        """
        Calculate and returns the agent's new position based on the chosen action.

        The elementary actions that the agent can perform are defined by integers from 0 to 3 as:
        - 0: Move up
        - 1: Move down
        - 2: Move left
        - 3: Move right
        """
        # Create a copy of the current agent position
        new_loc = self.agent_loc.copy()

        # Move the agent based on the action
        if action == 0:
            new_loc[0] += -1     # up
        elif action == 1:
            new_loc[0] += 1      # down
        elif action == 2:
            new_loc[1] += -1     # left
        elif action == 3:
            new_loc[1] += 1      # right

        return new_loc

    def apply_current(self):
        """
        Calculate and returns the agent's new position based on the current direction.

        Calculates and returns the agent's new position based on the current direction in the agent's
        current cell. Depending on the direction of the current, the agent may be moved up, down, left,
        or right. Some currents have two possible directions, in which case a random choice is made.

        In particular, the values from 4 to 6 of the current represent:
        - 4: Move down or left
        - 5: Move up or right
        - 6: Move down or right
        """
        # Save a copy of the current agent position
        new_loc = self.agent_loc.copy()
        current_dir = self.dir_grid[self.agent_loc[0], self.agent_loc[1]]  # current direction value in the agent's position cell

        # Sample the choice for the random action
        choice = random.randint(1, 2)

        # If the direction value is between 0 and 3 move the agent following elementary directions
        if current_dir <= 3:
            new_loc = self.move_agent(current_dir)

        # If the direction value is between 4 and 6 the agent follows the sampled current value
        elif current_dir == 4:
            if choice == 1:
                new_loc = self.move_agent(1)  # down
            else:
                new_loc = self.move_agent(2)  # left
        elif current_dir == 5:
            if choice == 1:
                new_loc = self.move_agent(0)  # up
            else:
                new_loc = self.move_agent(3)  # right
        elif current_dir == 6:
            if choice == 1:
                new_loc = self.move_agent(1)  # down
            else:
                new_loc = self.move_agent(3)  # right

        return new_loc

    def init_currents(self):
        """
        Sets up the direction of water currents within the environment's grid in 'dir_grid' attribute.
        The currents are set for corner cells, border cells, vortex areas, and middle areas of the grid.

        Currents are represented by integers indicating direction:
        - 0: up
        - 1: down
        - 2: left
        - 3: right
        - 4: down or left
        - 5: up or right
        - 6: down or right
        - 7: vortex center (no current)
        - 8: goal position (no current)
        """
        # Set currents in the corner cells (double currents + goal)
        self.dir_grid[0, 0] = 6
        self.dir_grid[0, self.grid_size - 1] = 4
        self.dir_grid[self.grid_size - 1, 0] = 5
        self.dir_grid[self.grid_size - 1, self.grid_size - 1] = 8

        # Set currents along the borders
        for i in range(1, self.grid_size - 1):
            # Bottom border (pushes the agent up)
            self.dir_grid[self.grid_size - 1, i] = 0

            # Top border (pushes the agent down)
            self.dir_grid[0, i] = 1

            # Right border (pushes the agent left )
            self.dir_grid[i, self.grid_size - 1] = 2

            # Left border (pushes the agent right)
            self.dir_grid[i, 0] = 3

        # Insert the initialized vortex currents within the grid
        for i in range(len(self.vortices)):
            start = self.start[i]
            end = start + self.v_size
            vortex = self.vortices[i]

            self.dir_grid[start:end, start:end] = vortex.v_dir

        # Set currents in middle areas (double currents)
        self.dir_grid[self.start[0]:self.v_size + 1, (self.start[0] + self.v_size):(self.grid_size - 1)] = 4
        self.dir_grid[self.start[1]:(self.grid_size - 1), (self.start[1] - self.v_size):self.v_size + 1] = 5

    def init_probs(self):
        """
        Sets the probability values in the `prob_grid` attribute.
        These values indicate the probability of the agent being affected by the water currents in each cell of the grid.
        The probabilities are set for the entire grid initially, and then specific values are assigned to
        the vortex areas.
        """
        # Initialize the entire grid with the default probability from the first cell of the first vortex
        self.prob_grid[:, :] = self.vortices[0].v_prob[0][0]

        # Set specific probabilities for the vortex areas
        for i in range(len(self.vortices)):
            start = self.start[i]
            end = start + self.v_size
            vortex = self.vortices[i]

            # Assign the probability matrix of the vortex to the corresponding area in the grid
            self.prob_grid[start:end, start:end] = vortex.v_prob

    # ------------------------------------------------ RENDERING METHODS ----------------------------------------------------

    def render_frame(self):
        """
        Creates a visual representation of the grid environment using the PyGame library.
        It initializes the PyGame window and clock if they are not already initialized, creates a canvas
        and draws each cell of the grid with the corresponding color based on the current state of the environment.
        """
        # Initialize PyGame window
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        # Initialize the PyGame clock
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a canvas for drawing the grid
        canvas = pygame.Surface((self.window_size, self.window_size))

        # Compute the dimension of a single cell in the grid
        cell_size = self.window_size // self.grid_size

        # Draw each cell of the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Get the color for the current cell
                color = self.colors_grid[x, y]

                # Draw the cell as a filled rectangle
                pygame.draw.rect(canvas, color, pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size))

                # Draw some borders for better visualization
                pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size), 1)

        # Update the PyGame display with the new canvas
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def update_render(self):
        """
        Assigns colors to different elements of the grid including open sea, goal, vortices, rocks, and the agent
        based on their current positions and the environment's configuration.
        """
        # Assign the color for the open sea to all cells in the grid
        self.colors_grid[:, :] = self.colors[0]

        # Assign the color for the goal to the goal location
        self.colors_grid[self.goal_loc[0], self.goal_loc[1]] = self.colors[2]

        # Assign colors for the vortices initialized in the respective class
        for i in range(len(self.vortices)):
            start = self.start[i]
            end = start + self.v_size
            vortex = self.vortices[i]

            self.colors_grid[start:end, start:end] = vortex.v_colors

        # Assign colors for the rocks if rocks are enabled in the environment
        if self.add_rocks:
            for i in range(0, len(self.rocks_pos)):
                self.colors_grid[self.rocks_pos[i][0], self.rocks_pos[i][1]] = self.colors[3]

        # Assign the color for the agent to the agent's current location
        self.colors_grid[self.agent_loc[0], self.agent_loc[1]] = self.colors[1]

    def close(self):
        """
        Ensures that if a Pygame window exists, it is properly closed, and the Pygame library is quit to free up resources.
        """
        # Check if the Pygame window is open
        if self.window is not None:
            # Quit the Pygame display
            pygame.display.quit()

            # Quit the Pygame library
            pygame.quit()

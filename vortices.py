import numpy as np


class Vortex:
    """
    The Vortex class simulates a vortex within a grid-based environment.
    Each vortex is defined by its dimensions and rotation direction, and it affects the probability, direction, and color
    of the grid cells it occupies.
    """
    def __init__(self, dim, rotation):
        """
        Initialize the Vortex object.
        Each vortex is described by its dimension, rotation verse, direction grid, probability grid and color grid.

        In particular, these attributes are described as:
        - dim: define the dimension of the square grid that represent the vortex
        - rotation: rotation verse of the vortex, can be clockwise or counterclockwise and directly influences the orientation
                    of its direction, probability and color grid
        - probability grid: It's a grid containing linear proportional increasing probabilities for the agent to be affected by
                            the current of the vortex
        - direction grid: It's a grid containing the direction of the currents for the considered vortex (rotation verse dependent)
        - color grid: It's a grid containing the colors of the vortex cells. Colours become darker going towards
                      the center of the vortex
        """
        self.dim = dim
        self.rotation = rotation
        self.min_prob = 0.05        # 5% probability for the first cell
        self.max_prob = 0.95        # 95% probability for the second last cell (small chance of escaping)
        self.v_prob, self.v_dir, self.v_colors = self.init_vortex()

    def init_vortex(self):
        """
        Initialize and returns the vortex probabilities, directions, and colors grids
        """
        # VORTEX PROBABILITY GRID
        # Set initial probabilities of the vortex as 0
        prob_m = np.zeros((self.dim, self.dim), dtype=np.float32)
        probabilities = np.linspace(self.min_prob, self.max_prob, (self.dim * self.dim) - 1)  # Linear samples probability values
        probabilities = np.append(probabilities, 0)  # Probability 0 for the vortex center

        # VORTEX COLORS GRID
        # Set initial colors of the vortex as 0
        colors_m = np.zeros((self.dim, self.dim, 3))

        # Obtain a color scale for vortex intensity
        start_color = np.array([32, 117, 213])
        end_color = np.array([0, 0, 139])
        steps = self.dim * self.dim

        # Calculate intermediate colors
        colors = [start_color + (end_color - start_color) * i / (steps - 1) for i in range(steps)]
        colors = np.clip(colors, 0, 255).astype(int)

        # VORTEX DIRECTIONS GRID
        # Set initial directions of the vortex as 0
        dir_m = np.zeros((self.dim, self.dim))

        # FILLING GRIDS PROCEDURE
        # Define parameters to iterate on the grid
        top, bottom, left, right = 0, self.dim - 1, 0, self.dim - 1    # Position indicators
        cur_el = 0                                                     # Current element
        tot_el = self.dim * self.dim                                   # Total elements

        # Based on the rotation verse of the vortex grids can be filled in a clockwise or counterclockwise order
        while top <= bottom and left <= right:
            # Clockwise filling
            if self.rotation == 'clockwise':
                # Fill the top side
                for i in range(left, right + 1):
                    prob_m[top][i] = probabilities[cur_el]  # Assign probability
                    if cur_el == tot_el - 1:
                        dir_m[top][i] = 7  # stop
                    elif i != right:
                        dir_m[top][i] = 3  # right
                    else:
                        dir_m[top][i] = 1  # down
                    colors_m[top][i] = colors[cur_el]       # Assign color
                    cur_el += 1                             # Proceed to next element
                top += 1

                # Fill the right side
                for i in range(top, bottom + 1):
                    prob_m[i][right] = probabilities[cur_el]
                    if cur_el == tot_el - 1:
                        dir_m[i][right] = 7  # stop
                    elif i != bottom:
                        dir_m[i][right] = 1  # down
                    else:
                        dir_m[i][right] = 2  # left
                    colors_m[i][right] = colors[cur_el]
                    cur_el += 1
                right -= 1

                # Fill the bottom side
                if top <= bottom:
                    for i in range(right, left - 1, -1):
                        prob_m[bottom][i] = probabilities[cur_el]
                        if cur_el == tot_el - 1:
                            dir_m[bottom][i] = 7  # stop
                        elif i != left:
                            dir_m[bottom][i] = 2  # left
                        else:
                            dir_m[bottom][i] = 0  # up
                        colors_m[bottom][i] = colors[cur_el]
                        cur_el += 1
                    bottom -= 1

                # Fill the left side
                if left <= right:
                    for i in range(bottom, top - 1, -1):
                        prob_m[i][left] = probabilities[cur_el]
                        if cur_el == tot_el - 1:
                            dir_m[i][left] = 7  # stop
                        elif i != top:
                            dir_m[i][left] = 0  # up
                        else:
                            dir_m[i][left] = 3  # right
                        colors_m[i][left] = colors[cur_el]
                        cur_el += 1
                    left += 1

            # Counterclockwise filling
            else:
                # Fill the top side
                for i in range(right, left - 1, -1):
                    prob_m[top][i] = probabilities[cur_el]
                    if cur_el == tot_el - 1:
                        dir_m[top][i] = 7  # stop
                    elif i != left:
                        dir_m[top][i] = 2  # left
                    else:
                        dir_m[top][i] = 1  # down
                    colors_m[top][i] = colors[cur_el]
                    cur_el += 1
                top += 1

                # Fill the left side
                for i in range(top, bottom + 1):
                    prob_m[i][left] = probabilities[cur_el]
                    if cur_el == tot_el - 1:
                        dir_m[i][left] = 7  # stop
                    elif i != bottom:
                        dir_m[i][left] = 1  # down
                    else:
                        dir_m[i][left] = 3  # right
                    colors_m[i][left] = colors[cur_el]
                    cur_el += 1
                left += 1

                # Fill the bottom side
                if top <= bottom:
                    for i in range(left, right + 1):
                        prob_m[bottom][i] = probabilities[cur_el]
                        if cur_el == tot_el - 1:
                            dir_m[bottom][i] = 7  # stop
                        elif i != right:
                            dir_m[bottom][i] = 3  # right
                        else:
                            dir_m[bottom][i] = 0  # up
                        colors_m[bottom][i] = colors[cur_el]
                        cur_el += 1
                    bottom -= 1

                # Fill the right side
                if left <= right:
                    for i in range(bottom, top - 1, -1):
                        prob_m[i][right] = probabilities[cur_el]
                        if cur_el == tot_el - 1:
                            dir_m[i][right] = 7  # stop
                        elif i != top:
                            dir_m[i][right] = 0  # up
                        else:
                            dir_m[i][right] = 2  # left
                        colors_m[i][right] = colors[cur_el]
                        cur_el += 1
                    right -= 1

        return prob_m, dir_m, colors_m

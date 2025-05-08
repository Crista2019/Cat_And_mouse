import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

class Gridworld(object):
    # takes in tuples of coord positions in (x,y) form and the number of obstacles we'd like to generate on the map
    def __init__(self, dimensions, cat_start, mouse_start, obstacles=None):
        # width and height of the track
        self.left_border = 0
        self.right_border = dimensions[0] - 1
        self.top_border = 0
        self.bottom_border = dimensions[1] - 1
        # tuple of cat
        self.cat_start = cat_start
        self.mouse_start = mouse_start

        # initialize grid of proper dimensions (set all to 0 to start)
        # zero will represent the walls and obstacles of the room
        self.grid = np.zeros((self.bottom_border, self.right_border))

        # one represents traversable area
        self.grid[1:self.bottom_border-1, 1:self.right_border-1] = 1

        # setting cat position
        self.grid[cat_start[1]][cat_start[0]] = 2

        # setting mouse position
        self.grid[mouse_start[1]][mouse_start[0]] = 3

        self.obstacle_coords = []

        # if there are obstacles (such as furniture, etc), these too are nontraversable
        if obstacles is not None:
            for _ in range(obstacles):
                valid_obstacle = False
                while not valid_obstacle:
                    # the obstacle cannot be where the cat and mouse begin
                    obstacle = (random.randint(1, self.right_border-1), random.randint(1, self.bottom_border-1))
                    if obstacle != self.cat_start:
                        if obstacle != self.mouse_start:
                            valid_obstacle = True
                            self.obstacle_coords.append((obstacle[1],obstacle[0]))
        for o in self.obstacle_coords:
            self.grid[o[1],o[0]] = 4


    def reset(self):
        self.grid[np.nonzero(self.grid)] = 1
        # add back obstacles
        for o in self.obstacle_coords:
            self.grid[o[1], o[0]] = 4
        # setting cat position
        self.grid[self.cat_start[1]][self.cat_start[0]] = 2
        # setting mouse position
        self.grid[self.mouse_start[1]][self.mouse_start[0]] = 3

    def evaluate(self, x, y):
        # returns the value of a valid space
        if self.is_off_limits(x, y):
            return "Error not accessible to agent"
        return self.grid[y, x]

    def is_off_limits(self, x, y):
        # respond true if the proposed x,y position of the grid is on the finish line
        if x >= self.right_border or x < self.left_border:
            return True
        elif y >= self.bottom_border or y < self.top_border:
            return True
        elif self.grid[y, x] == 0 or self.grid[y, x] == 4:
            return True
        return False

    def __str__(self):
        return str(self.grid)

    # visualize the gridworld
    def visualize(self):
        fig, ax = plt.subplots()
        cmap = colors.ListedColormap(['#4d4d9fff', '#ba6833ff', '#ff812dff', '#ac9d93ff', '#4d4d9fff'])
        ax.imshow(self.grid, cmap=cmap)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        return plt.show()
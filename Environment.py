import numpy as np

class Environment(object):
    # the session terminates if:
    # 1) the cat catches the mouse
    # 2) the mouse hits a wall or obstacle

    def __init__(self, gridworld, cat, mouse):
        self.gridworld = gridworld
        self.cat = cat
        self.mouse = mouse
        # the row and column indices for the nonzero grid spaces
        self.full_state_space = [tuple(i) for i in np.transpose(np.nonzero(self.gridworld.grid))]

    def cat_reward(self, prev_pos, new_pos, tired_factor):
        # if the cat gets tired, we are done (to avoid infinite play, which is ideal but not realistic)
        if tired_factor >= 100:
            return "done"

        # if this is the mouse (3), get big reward
        if self.gridworld.evaluate(new_pos[0],new_pos[1]) == 3:
            return np.random.normal(10, 0.25)

        # if this is near the mouse (3)
        # checks if the mouse is anywhere in the 8 (max) surrounding squares
        # because of the boundary at the walls, the agent can never look out of bounds
        for i,j in [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1,1), (-1,-1), (1,-1)]:
            if self.gridworld.evaluate(new_pos[0]+i,new_pos[1]+j) == 3:
                # less bored with interaction
                self.cat.tired_factor -= 1
                return np.random.normal(3, 0.25)

        # cat doesn't want to hit obstacles either
        if self.gridworld.is_off_limits(new_pos[0],new_pos[1]):
            # running into things hurts
            return np.random.normal(-3, 0.25)

        if prev_pos == new_pos:
            # not moving makes the cat more bored
            self.cat.tired_factor += 3
            return np.random.normal(-3, 0.25)

        # if this is anything else, slight negative cost
        self.cat.tired_factor += 1
        return np.random.normal(-0.5, 0.25)

    def mouse_reward(self, prev_pos, new_pos):
        # if this is the cat (2), we get big neg reward
        if self.gridworld.evaluate(new_pos[0],new_pos[1]) == 2:
            return np.random.normal(-3, 0.25)

        # if this is an obstacle or wall (0), we die
        if self.gridworld.is_off_limits(new_pos[0],new_pos[1]):
            return "done"

        if prev_pos == new_pos:
            # mouse toy shouldn't ever stop
            return np.random.normal(-3, 0.25)

        # if we are in the vicinity of the cat (2 within the 8 blocks surrounding mouse), give higher reward
        for i,j in [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1,1), (-1,-1), (1,-1)]:
            if self.gridworld.evaluate(new_pos[0]+i,new_pos[1]+j) == 2:
                return np.random.normal(-5, 0.25)

        # if we are at nothing (open floor) add a minimal reward, noise
        return np.random.normal(0.5, 0.25)

    def run(self, cat_action, mouse_action):
        # runs one step of the simulation
        # takes in the cat_action and mouse action
        # returns the cat position and reward and mouse position and reward (respectively) after action

        # CAT TURN
        # take action, returns new position
        old_cat_pos = self.cat.position
        new_cat_pos = self.cat.take_turn(cat_action)

        cat_reward = self.cat_reward(old_cat_pos, new_cat_pos, tired_factor=self.cat.tired_factor)

        # if this is a valid move, we keep going
        # otherwise the cat doesn't move (we assume it hits the wall)
        if self.gridworld.is_off_limits(*new_cat_pos):
            self.cat.position = old_cat_pos
            returned_cat_pos = old_cat_pos
        else:
            self.cat.position = new_cat_pos
            returned_cat_pos = new_cat_pos

        # MOUSE TURN
        old_mouse_pos = self.mouse.position
        new_mouse_pos = self.mouse.take_turn(mouse_action)

        mouse_reward = self.mouse_reward(old_mouse_pos, new_cat_pos)

        # if this is a valid move, we keep going
        # if the move was invalid, we stay where we were
        if self.gridworld.is_off_limits(*new_mouse_pos):
            self.mouse.position = old_mouse_pos
            returned_mouse_pos = old_mouse_pos
        else:
            self.mouse.position = new_mouse_pos
            returned_mouse_pos = new_mouse_pos

        # update the grid world
        new_grid = self.gridworld.grid
        new_grid[np.nonzero(self.gridworld.grid)] = 1
        new_grid[returned_cat_pos] = 2
        new_grid[returned_mouse_pos] = 3

        self.gridworld.grid = new_grid

        return returned_cat_pos, cat_reward, returned_mouse_pos, mouse_reward, new_grid
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

    def cat_reward(self, new_pos):
        # if this is the mouse (3), we terminate the game
        if self.gridworld.evaluate(new_pos[0],new_pos[1]) == 3:
            return "done"

        # if this is near the mouse (3)
        # checks if the mouse is anywhere in the 8 (max) surrounding squares
        # because of the boundary at the walls, the agent can never look out of bounds
        for i,j in [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1,1), (-1,-1), (1,-1)]:
            if self.gridworld.evaluate(new_pos[0]+i,new_pos[1]+j) == 3:
                return np.random.normal(3, 0.25)


        # if this is anything else, slight negative cost
        return np.random.normal(-0.5, 0.25)

    def mouse_reward(self, new_pos):
        # if this is the cat (2), we terminate the game
        if self.gridworld.evaluate(new_pos[0],new_pos[1]) == 2:
            return "done"

        # if this is an obstacle or wall (0), we get a negative reward
        if self.gridworld.evaluate(new_pos[0],new_pos[1]) == 0:
            return np.random.normal(-3, 0.25)

        # if we are in the vicinity of the cat (2 within the 8 blocks surrounding mouse), give higher reward
        for i,j in [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1,1), (-1,-1), (1,-1)]:
            if self.gridworld.evaluate(new_pos[0]+i,new_pos[1]+j) == 2:
                return np.random.normal(3, 0.25)

        # if we are at nothing (open floor) add a minimal reward, noise
        return np.random.normal(0.5, 0.25)

    def run(self, cat_action, mouse_action):
        # runs one step of the simulation
        # takes in the cat_action and mouse action
        # returns the cat position and reward and mouse position and reward (respectively) after action

        # CAT TURN
        # take action, returns new position
        new_cat_pos = self.cat.take_turn(cat_action)
        # if this is a valid move, we keep going
        # otherwise the cat doesn't move (we assume it hits the wall but we don't penalize)
        if not self.gridworld.is_off_limits(*new_cat_pos):
            self.cat.position = new_cat_pos
        else:
            # keep old pos
            new_cat_pos = self.cat.position

        cat_reward = self.cat_reward(new_cat_pos)

        # MOUSE TURN
        new_mouse_pos = self.mouse.take_turn(mouse_action)
        # if this is a valid move, we keep going
        if not self.gridworld.is_off_limits(*new_mouse_pos):
            self.mouse.position = new_mouse_pos
            mouse_reward = self.cat_reward(new_cat_pos)
        else:
            # we want the negative reward for trying to hit the wall or obstacle
            mouse_reward = self.cat_reward(new_cat_pos)
            # if the move was invalid, we stay where we were
            new_mouse_pos = self.mouse.position

        return new_cat_pos, cat_reward, new_mouse_pos, mouse_reward
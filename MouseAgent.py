from itertools import product

def valid_velocity(vel):
    # we will assume the mouse is faster than the cat
    # valid velocity components are positive and less than 5
    # we never want the mouse to stop moving!
    if 0 <= vel < 5:
        return True
    return False

class MouseAgent:
    # the mouse agent is the toy
    # the goal of the mouse agent is to tease the cat
    # this means that it will be rewarded for proximity to the cat
    # we assume the toy mouse is charged and has infinite energy, so there is no cost of motion
    def __init__(self, pos):
        # tuple representing the x, y position of the mouse at the start of an episode
        self.position = pos

        # combination of all possible changes to vertical and horizontal velocity
        # there are nine possible actions for changing x and y by -1,0,1
        self.action_space = list(product(list(product([-1, 0, 1], ["x"])), list(product([-1, 0, 1], ["y"]))))

        # how many squares to move up and right in a turn
        self.x_velocity = 0
        self.y_velocity = 0

    def reset(self):
        self.x_velocity = 0
        self.y_velocity = 0

    def update_velocity(self, action):
        # updates based on the action taken
        # in form ((#int, 'x'), (#int, 'y'))
        # going right is positive x velocity and going left is negative
        x_change = action[0][0] + self.x_velocity
        # going up corresponds to negative y velocity
        y_change = - action[1][0] + self.y_velocity

        # update the velocities to desired action if possible
        if valid_velocity(x_change):
            new_x_vel = x_change
        else:
            new_x_vel = self.x_velocity

        # reverse direction here for the check as up is negative in this implementation
        if valid_velocity(-y_change):
            new_y_vel = y_change
        else:
            new_y_vel = self.y_velocity

        return new_x_vel, new_y_vel

    def take_turn(self, next_action):
        # this is when the agent selects the next action
        updated_velocities = self.update_velocity(self.action_space[next_action])

        self.x_velocity = updated_velocities[0]
        self.y_velocity = updated_velocities[1]
        # returns the x, y position of the next action
        return self.position[0] + updated_velocities[0], self.position[1] + updated_velocities[1]
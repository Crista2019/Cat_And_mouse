# local imports
from Gridworld import Gridworld
from Environment import Environment
from CatAgent import CatAgent
from MouseAgent import MouseAgent
# math tools
import numpy as np
from itertools import product
import random
# visualization tools
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def grids_to_video(grids, output_file='output.mp4', fps=5):
    # transform list of grids (gridworld.grid arrays) into a video file
    fig, ax = plt.subplots()
    im = ax.imshow(grids[0], animated=True)

    def update_fig(i):
        im.set_array(grids[i])
        return im,

    ani = animation.FuncAnimation(fig, update_fig, frames=len(grids), interval=1000/fps, blit=True)
    ani.save(output_file, writer='ffmpeg', fps=fps)
    plt.close(fig)

def control_func(environment, n, discount_factor=0.99, epsilon=0.1):
    # implements Monte Carlo control
    # reverse so these are x,y pairs of all possible positions that can be occupied
    state_space = [(s[1], s[0]) for s in env.full_state_space]

    # state action combos for our 9 different velocity change actions
    num_actions = 9
    action_states = list(product(state_space, np.arange(num_actions)))

    # separate Q tables for cat and mouse
    q_table_cat = {s: [0.0] * num_actions for s in state_space}
    q_table_mouse = {s: [0.0] * num_actions for s in state_space}

    returns_cat = {sa: [] for sa in action_states}
    returns_mouse = {sa: [] for sa in action_states}

    policy_cat = {s: [1.0 / num_actions] * num_actions for s in state_space}
    policy_mouse = {s: [1.0 / num_actions] * num_actions for s in state_space}

    for _ in range(n):
        # run one episode to obtain the state/action/reward combo
        episode = run_episode(policy_cat, policy_mouse, environment)
        # update our q table for cat and mouse
        # make sure to cut the very last episode trial because the reward is "done"
        q_table_cat, new_returns_cat = update_q_table(policy_cat, returns_cat, episode[0][:-1], discount_factor)
        q_table_mouse, new_returns_mouse = update_q_table(policy_mouse, returns_mouse, episode[1][:-1], discount_factor)

        # use q values to update our policy
        policy_cat = update_policy(policy_cat, state_space, num_actions, q_table_cat, epsilon)
        policy_mouse = update_policy(policy_mouse, state_space, num_actions, q_table_mouse, epsilon)

    return policy_cat, q_table_cat, policy_mouse, q_table_mouse

def update_q_table(q_values, returns, episode, discount_factor):
    new_q_values = q_values
    new_returns = returns

    # moving backward through episode to apply temporal discount
    G = 0
    for state, action, reward in reversed(episode):
        G = discount_factor * G + reward
        new_returns[(state, action)].append(G)
        new_q_values[state][action] = np.mean(new_returns[(state, action)])

    return new_q_values, new_returns


def update_policy(policy, state_space, num_actions, q_table, epsilon):
    new_policy = policy
    for s in state_space:
        # # reverse the y,x -> x,y
        # get the index of the "best" action
        optimal_action = np.argmax(q_table[s])
        for a in range(num_actions):
            if optimal_action == a:
                new_policy[s][a] = 1 - epsilon + epsilon / num_actions
            else:
                new_policy[s][a] = epsilon / num_actions
    return new_policy

def run_episode(cat_policy, mouse_policy, environment):
    terminal = False
    episode_t = [[],[]]

    cat_probabilities = cat_policy[cat_start_pos]
    mouse_probabilities = mouse_policy[mouse_start_pos]

    while not terminal:
        print('trial:', len(episode_t[0]))
        # get the probabilities for acting based on policies given the states of our agents
        # use this to select the action
        cat_action = np.random.choice(range(len(cat_probabilities)), size=1, p=cat_probabilities)[0]
        mouse_action = np.random.choice(range(len(mouse_probabilities)), size=1, p=mouse_probabilities)[0]

        new_cat_pos, cat_reward, new_mouse_pos, mouse_reward = environment.run(cat_action, mouse_action)

        if cat_reward == "done" or mouse_reward == "done":
            terminal = True

        cat_probabilities = cat_policy[new_cat_pos]
        mouse_probabilities = mouse_policy[new_mouse_pos]

        episode_t[0].append((new_cat_pos, cat_action, cat_reward))
        episode_t[1].append((new_mouse_pos, mouse_action, mouse_reward))

    return episode_t

if __name__ == '__main__':
    # set the locations where the cat and mouse should start
    # these can be randomly generated or chosen deliberately
    cat_start_pos = (9, 4)
    mouse_start_pos = (10, 5)

    g = Gridworld(dimensions=(20, 20), cat_start=cat_start_pos, mouse_start=mouse_start_pos, obstacles=0)

    # visualize the track in grid space
    g.visualize()

    # create the two agents
    cat = CatAgent(pos=cat_start_pos)
    mouse = MouseAgent(pos=mouse_start_pos)

    # create environment
    env = Environment(g, cat, mouse)

    # get the agents to actually do stuff
    control_func(env, n=1)

    # visualize the run
    # grids = None
    # grids_to_video(grids, output_file='random_grids.mp4', fps=5)


from Gridworld import Gridworld
from Environment import Environment
from CatAgent import CatAgent
from MouseAgent import MouseAgent
import numpy as np
from itertools import product
import random

def control_func(environment, n=1000, discount_factor=0.99, epsilon=0.1):
    # implements Monte Carlo control
    # reverse so these are x,y pairs
    state_space = [(s[1], s[0]) for s in env.full_state_space]

    # state action combos for our 9 different velocity change actions
    num_actions = 9
    action_states = list(product(state_space, np.arange(num_actions)))


    # separate Q tables for cat and mouse
    q_table_cat = {s: [0.0] * num_actions for s in state_space}
    q_table_mouse = {s: [0.0] * num_actions for s in state_space}

    # policies for our cat and mouse agents defining their behavior
    # (define them as simple greedy policies for now)
    # we want our cat to chase the mouse, its going to be super greedy
    def policy_cat(tired_threshold=40,chase_counter=0):
        # some random actions to make cat silly
        if random.uniform(0, 1) < epsilon:
            return random.choice(CatAgent.action_space), chase_counter  # Explore
        # cat is done playing when it gets tired
        elif chase_counter >= tired_threshold:
            CatAgent.x_velocity = 0
            CatAgent.y_velocity = 0
            return (0,0,0)
        else:
            return np.argmax(q_table_cat[CatAgent.pos]), chase_counter # Exploit
    
    # policy for mouse agent
    # we want the mouse to avoid cat and also be random, it be more exploratory
    def policy_mouse():
        if random.uniform(0, 1) < epsilon/2:
            return random.choice(MouseAgent.action_space)  # Explore
        else:
            return np.argmax(q_table_mouse[MouseAgent.pos])  # Exploit

    returns_cat = {sa: [] for sa in action_states}
    returns_mouse = {sa: [] for sa in action_states}

    policy_cat = {s: [1.0 / num_actions] * num_actions for s in state_space}
    policy_mouse = {s: [1.0 / num_actions] * num_actions for s in state_space}

    for _ in range(n):
        # run one episode to obtain the state/action/reward combo
        episode = run_episode(policy_cat, policy_mouse, environment)
        # update our q table for cat and mouse
        q_table_cat = update_q_table(policy_cat, returns_cat, episode[0], discount_factor)
        q_table_mouse = update_q_table(policy_mouse, returns_mouse, episode[1], discount_factor)

        # use q values to update our policy
        policy_cat = update_policy(policy_cat, state_space, num_actions, q_table_cat, epsilon)
        policy_mouse = update_policy(policy_mouse, state_space, num_actions, q_table_mouse, epsilon)

    return policy_cat, q_table_cat, policy_mouse, q_table_mouse

def update_q_table(q_values, returns, episode, discount_factor):
    pass

def update_policy(policy, state_space, num_actions, q_table, epsilon):
    pass

def run_episode(cat_policy, mouse_policy, environment):
    terminal = False
    episode_t = [[],[]]


    cat_probabilities = cat_policy[cat_start_pos]
    mouse_probabilities = mouse_policy[mouse_start_pos]

    while not terminal:
        # get the probabilities for acting based on policies given the states of our agents
        # use this to select the action

        cat_action = np.random.choice(range(len(cat_probabilities)), size=1, p=cat_probabilities)[0]
        mouse_action = np.random.choice(range(len(mouse_probabilities)), size=1, p=mouse_probabilities)[0]

        new_cat_pos, cat_reward, new_mouse_pos, mouse_reward = environment.run(cat_action, mouse_action)

        if cat_reward == "done" or mouse_reward == "done":
            terminal = True

        cat_state = new_cat_pos
        mouse_state = new_mouse_pos

        cat_probabilities = cat_policy[cat_state]
        mouse_probabilities = mouse_policy[mouse_state]

        episode_t[0].append((new_cat_pos, cat_action, cat_reward))
        episode_t[1].append((new_mouse_pos, mouse_action, mouse_reward))

    return episode_t

if __name__ == '__main__':
    # set the locations where the cat and mouse should start
    # these can be randomly generated or chosen deliberately
    cat_start_pos = (3, 4)
    mouse_start_pos = (5, 6)

    g = Gridworld(dimensions=(20, 20), cat_start=cat_start_pos, mouse_start=mouse_start_pos, obstacles=10)

    # visualize the track in grid space
    g.visualize()

    # create the two agents
    cat_agent = CatAgent(pos=cat_start_pos)
    mouse_agent = MouseAgent(pos=mouse_start_pos)

    # create environment
    env = Environment(g, cat_agent, mouse_agent)

    # get the agents to actually do stuff


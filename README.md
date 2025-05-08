# Final Project: Game of Cat and Mouse

For stereotypically finicky pets, like cats, interest in a new toy is not guaranteed, nor is prolonged play. 
Interactive toys, famously the self-moving ball Cheerble, seek to address this issue by offering a "smart" solution to engaging one's pet. 
However, no cat is the same and one toy's "smart" behavior does not account for the variety of play styles cats can exhibit. 
Thus, we propose a new interactive cat toy which utilizes "cat-in-the-loop" feedback to personalize its engagement and evasion behavior to suits each cat's long-term enrichment needs. 
In this project, we explore the dynamics at play in "cat and mouse" interactions, in which one agent (the cat) aims to capture, and the other (the toy) intends to tease while avoiding capture. 
Using virtual 2D simulations, we trained a toy agent to interact with a cat agent without terminating the play session by getting caught.

## Model Overview
We use a gridworld to simulate a typical living room or cat-play space. 
Points on the grid represent the cat position, the mouse toy position, obstacles such as furniture, and walls to avoid colliding with.

### Reward is given to the cat if
1. the cat catches the mouse
2. the cat moves to a space adjacent to the mouse
3. slight negative cost of movement

### Reward is given to the mouse if
1. the mouse is in the vicinity of the cat
2. negative reward for hitting obstacles or walls

### The session terminates if:
1. the cat catches the mouse
2. the mouse hits a wall or obstacle

# Set Up Instructions 
```
### installing necessary packages
# math tools
!pip install numpy
!pip install itertools
!pip install random
# visualization tools
!pip install opencv-python
!brew install ffmpeg
!pip install copy
!pip install matplotlib
!pip install pandas
```

# Usage
```
# define initial position of agents
cat_start_pos = (11, 4)
mouse_start_pos = (10, 5)

# generate a grid world
g = Gridworld(dimensions=(20, 20), cat_start=cat_start_pos, mouse_start=mouse_start_pos, obstacles=10)

# visualize the track in grid space
g.visualize()

# create the two agents
cat = CatAgent(pos=cat_start_pos)
mouse = MouseAgent(pos=mouse_start_pos)

# create environment
env = Environment(g, cat, mouse)

# get the agents to actually act, returns the policy (probabilities), q tables, and saved states
policy_cat, q_table_cat, policy_mouse, q_table_mouse, all_episodes, all_grids = control_func(env, n=100)
```

# Presentation!
[Game of Cat and Mouse](https://youtu.be/ThAuTyec2oY)

import gym
import numpy as np

env = gym.make("MountainCar-v0")

'Determine q-table shape'
DISCRETE_OBS_SPACE_SIZE = [20, 20] # 20 bins of vertical lines, 20 bins of horizontal lines
# DISCRETE_OBS_SPACE_SIZE = [20] * len(env.observation_space.high) # for a general environment with more than 2 bins
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SPACE_SIZE #how many things will go into each bin of the Q learning table

'Shape the reward'
LEARNING_RATE = 0.1
GAMMA = 0.95  # discount, defines importance of future actions
EPISODES = 25000  # population size

'Introduce randomness'
epsilon = 0.5  # amount of randomness of actions [0, 1] - 1 gives high exploration and randomness
START_EPSILON_DECAY = 200  # start decaying from first episode
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SPACE_SIZE + [env.action_space.n]))  # since the reward is always -1 till reaches flag, then 0

def discretize_state(continuous_state):
    discrete_state = (continuous_state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    if episode % 1000 == 0:  # render every 1000 episodes
        print(episode)
        render = True
    else:
        render = False
    discrete_state = discretize_state(env.reset())  # initial state
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])  # can either be 0 - move left, 1- nothing, 2 - move right
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = discretize_state(new_state)
        if render:
            env.render()  # draw the environment
        if not done:  # didn't reach flag or finish 200 steps
            current_q = q_table[discrete_state][action]
            max_future_q = np.max(q_table[new_discrete_state])  # choose the action that benefits me most from this position

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + GAMMA * max_future_q)
            q_table[discrete_state][action] = new_q
        elif new_state[0] >= env.goal_position:  # reached flag
            q_table[discrete_state][action] = 0  # no more punishment

        discrete_state = new_discrete_state

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        pass
    else:
        epsilon -= epsilon_decay_value



env.close()

import time
import gym
import gym_maze
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_reward(next_state):
    goal = 1
    step_penalty = -0.01

    if next_state == 99:
        return goal
    else:
        return step_penalty


def epsilon_greedy(state):
    if np.random.random() < epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(4)


def q_learning(curr_state, next_state, curr_action, Q):

    temporal_difference = get_reward(next_state) + (gamma * np.max(Q[next_state]))\
                          - Q[curr_state][curr_action]

    Q[curr_state][curr_action] = Q[curr_state][curr_action] + (alpha * temporal_difference)
    return Q


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    env.reset()

    # Define the maximum number of iterations
    NUM_EPISODES = 1000
    MAX_STEP = 100
    epsilon = 0.1
    gamma = 0.9
    alpha = 0.9
    done = False

    Q = np.zeros((100, 4))
    current_state = 0
    current_action = epsilon_greedy(0)

    finding_count = 0
    converged = False

    total_reward_list = []
    mean_list = []
    x_axis = []
    for i in range(1000):
        x_axis.append(i)

    for episode in range(NUM_EPISODES):
        total_reward = 0
        print(f"episode: {episode}")
        old_q = np.copy(Q)
        for step in range(MAX_STEP):
            env.render()

            next_state, reward, done, truncated = env.step(current_action)

            total_reward += reward

            next_state = int(next_state[1] * 10 + next_state[0])

            next_action = epsilon_greedy(next_state)

            Q = q_learning(current_state, next_state, current_action, Q)

            if episode == 30:
                epsilon = 0.3
            elif episode == 50:
                epsilon = 0.6
            elif episode == 70:
                epsilon = 0.9
            elif episode == 90:
                epsilon = 1

            current_state = next_state
            current_action = next_action

            if done or truncated:
                finding_count += 1
                print(f"Reached the goal for the {finding_count} time")
                env.reset()
                current_state = 0
                current_action = epsilon_greedy(current_state)
                break

        env.reset()
        current_state = 0
        current_action = epsilon_greedy(current_state)

        if abs(np.linalg.norm(old_q) - np.linalg.norm(Q)) < 1e-4:
            print(f"in episode: {episode} converged")

        total_reward_list.append(total_reward)

        mean = 0
        sum = 0
        for t_reward in total_reward_list:
            sum += t_reward
        mean = sum / len(total_reward_list)
        mean_list.append(mean)

    plt.plot(x_axis, mean_list)
    plt.xlabel("Episode")
    plt.ylabel("Mean Total Reward")
    plt.show()

    # Close the environment
    env.close()

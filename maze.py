import time
import gym
import gym_maze
import numpy as np


def epsilon_greedy(state):
    if np.random.random() < epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(4)


def q_learning(curr_state, next_state, reward, curr_action, Q):
    temporal_difference = reward + (gamma * np.max(Q[next_state])) - Q[curr_state][curr_action]
    Q[curr_state][curr_action] = Q[curr_state][curr_action] + (alpha * temporal_difference)
    return Q


if __name__ == '__main__':

    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    # Define the maximum number of iterations
    NUM_EPISODES = 1000
    MAX_STEP = 99
    epsilon = 0.1
    gamma = 0.9
    alpha = 0.9
    done = False

    Q = np.zeros((100, 4))
    current_state = 0
    current_action = epsilon_greedy(0)

    finding_count = 0
    converged = False

    q_list = []

    for episode in range(NUM_EPISODES):
        old_q = np.copy(Q)
        for step in range(MAX_STEP):
            env.render()

            next_state, reward, done, truncated = env.step(current_action)

            # if episode > 300:
            #     time.sleep(0.1)

            next_state = next_state[1]*10 + next_state[0]

            next_action = epsilon_greedy(next_state)

            if not converged:
                Q = q_learning(current_state, next_state, reward, current_action, Q)

            if episode == 30:
                epsilon = 0.3
            elif episode == 60:
                epsilon = 0.6
            elif episode == 90:
                epsilon = 0.9
            elif episode == 120:
                epsilon = 0.99

            current_state = next_state
            current_action = next_action

            if done or truncated:
                finding_count += 1
                #print(f"Reached the goal for the {finding_count} time")
                observation = env.reset()
                current_state = 0
                current_action = epsilon_greedy(current_state)
                break

        observation = env.reset()
        current_state = 0
        current_action = epsilon_greedy(current_state)

        # if episode == 120:
        #     if np.max(np.abs(q_list[0], q_list[1])) < 1e-10 :
        #         print(f"in episode: {episode} converged")

        if np.allclose(old_q, Q, rtol=0.0001, atol=0.0001):
            print(f"in episode: {episode} converged")
            converged = True
    # Close the environment
    env.close()

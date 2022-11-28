import gym
from gym_minigrid.wrappers import *
import time
import pickle
from os.path import exists
import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import logger
from utils import read_yaml, timer
from typing import Dict, List, Tuple

logger = logger.get_logger(__name__)

def reshape_state(state: np.ndarray) -> np.ndarray:
    (rows, cols, x) = state.shape
    _ = np.reshape(state, [rows*cols*x, 1], 'F')[0:rows*cols]
    state_reshaped = np.reshape(_, [rows, cols], 'F')
    return state_reshaped


def generate_state_hash(arr: np.ndarray) -> int:
    arr_enc = str(arr).encode()
    return hashlib.md5(arr_enc).hexdigest()


def initialise_env(config) -> np.ndarray:
    env = gym.make(config['general']['ENV_NAME'])
    env = ImgObsWrapper(env)
    return env


def initialise_state(env):
    _ = env.reset()
    _state = reshape_state(_)
    state = generate_state_hash(_state)
    return state


def take_random_action(config):
    """
    left = 0
    right = 1
    forward = 2
    """
    return np.random.randint(0, config['general']['possible_actions'])


def take_action_epsilon_greedy(q_table, state, config, epsilon):
    if (np.random.random() < epsilon):
        action = take_random_action(config)
    else:
        action = np.argmax(q_table[state])
    logger.debug(f"Action chosen: {action}")
    return action


def initialise_q(config, state):
    q = {}
    val = np.random.uniform(0, 1, config['general']['possible_actions'])
    q[state] = val
    return q


def create_grid(config):
    grid = {}
    key = 0
    for i in range(len(config['grid_search']['alpha_range'])):
        for j in range(len(config['grid_search']['gamma_range'])):
            grid[key] = {'alpha': config['grid_search']['alpha_range'][i],
                         'gamma': config['grid_search']['gamma_range'][j]}
            key = key + 1
    logger.info(f"Grid: {grid}")
    return grid


def update_epsilon(config, prev_epsilon: float, episode: int) -> float:
    """

    """
    if prev_epsilon > config['grid_search']['min_epsilon']:
        epsilon = prev_epsilon * config['grid_search']['epsilon_decay_rate']
    else:
        epsilon = config['grid_search']['min_epsilon']
    return epsilon


@timer
def train(config, hparams, algo: str, render=False):
    env = initialise_env(config)
    state = initialise_state(env)
    q = initialise_q(config, state)
    rewards = []
    step_tracker = []
    prev_epsilon = config['grid_search']['max_epsilon']  # Initalise epsilon to max_value
    for episode in range(config['general']['episodes']):
        state = initialise_state(env)
        epsilon = update_epsilon(config, prev_epsilon, episode)
        for step in range(0, config['general']['max_steps']):
            action = take_action_epsilon_greedy(q, state, config, epsilon)
            _next_state, reward, done, info = env.step(action)
            if render:
                env.render()
                time.sleep(0.05)
            if done:
                if reward > 0:
                    step_tracker.append(step)
                    logger.debug(
                        f"Finished episode successfully taking {step} steps and receiving reward {reward}")
                else:
                    pass
                break
            _next_state = reshape_state(_next_state)
            next_state = generate_state_hash(_next_state)
            if q.get(next_state) is None:
                val = val = np.random.uniform(0, 1, config['general']['possible_actions'])
                q[next_state] = val
            if algo == 'ql':
                next_action = np.argmax(q[next_state])
                q[state][action] = q[state][action] + hparams['alpha'] * (reward + hparams['gamma'] * (q[next_state][next_action]) - q[state][action])
            elif algo == 'sarsa':
                next_action = take_action_epsilon_greedy(q, state, config, epsilon) # Choose next_action from next_state using epsilon greedy with Q
                q[state][action] = q[state][action] + hparams['alpha'] * (reward + hparams['gamma'] * (q[next_state][next_action]) - q[state][action])#update q
            state = next_state
        rewards.append(reward)
        prev_epsilon = epsilon  # Update epsilon so that it can decay
    agent_name = f'{algo}_agent_{int(time.time())}'
    _save_agent(config, q, agent_name)
    plot_reward(config, rewards, agent_name)
    result = {'alpha': hparams['alpha'],
              'gamma': hparams['gamma'],
              'min_epsilon': config['grid_search']['min_epsilon'],
              'max_epsilon': config['grid_search']['max_epsilon'],
              'epsilon_decay_rate': config['grid_search']['epsilon_decay_rate'],
              'avg_reward': np.sum(rewards) / np.count_nonzero(rewards),
              'completion_rate': np.count_nonzero(rewards) / len(rewards),
              'avg_steps': np.sum(step_tracker) / len(step_tracker)}
    return agent_name, result


def _save_agent(config, agent, agent_name) -> None:
    """
        Dumps python dict into a pickle file and saves to specified path in config.

        Args:
            config (dict) : Config settings and variables
            agent (dict): Q table
            agent_name (str) : Unique name per agent.
    """
    with open(f"{config['general']['model_dir']}/{agent_name}.pickle", 'wb') as handle:
        pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Q table saved successfully.")
        handle.close()


def plot_reward(config, rewards, agent_name) -> None:
    """
        Plots the cumulative sum of the rewards divided by the number of episodes and saves image to location specified in config.

        Args:
            config (dict) : Config settings and variables
            rewards (list) : Rewards accumulated at each episode
            agent_name (str) : Unique name per agent.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_title(fr"Accumalated Reward averaged over time")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accumulated Reward')
    x = [i+1 for i in range(len(rewards))]
    acc_rewards = np.cumsum(rewards)
    avg_rewards = [acc_rewards[i]/(i+1) for i in range(len(acc_rewards))]
    plt.yticks(np.arange(min(avg_rewards)-0.1, max(avg_rewards)+0.1, 0.1))
    ax.plot(x, avg_rewards)
    plt.savefig(f"{config['general']['perf_art_dir']}/reward_{agent_name}.png")
    logger.info(f"Rewards plot saved successfully.")


@timer
def write_results(config, results, algo) -> None:
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(f"{config['general']['gs_results']}/{algo}_gs_results_{int(time.time())}.csv")
    logger.info(f"Grid search results csv saved successfully.")

@timer
def main(config_path):
    config = read_yaml(config_path)
    grid = create_grid(config)
    algos = config['general']['algos']
    for algo in algos:
        results = {}
        for key in grid.keys():
            hparams = grid[key]
            agent_name, result = train(config, hparams, algo)
            results[agent_name] = result
        write_results(config, results, algo)


if __name__ == '__main__':
    config_path = "../conf/config.yaml"
    main(config_path)
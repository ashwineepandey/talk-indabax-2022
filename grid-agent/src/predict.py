import gym
from gym_minigrid.wrappers import *
import time
import pickle
from os.path import exists
import numpy as np
import hashlib
import argparse
import matplotlib.pyplot as plt
import logger
from utils import read_yaml, timer, debug
from typing import Dict, List, Tuple, Union
from train_gridsearch import reshape_state, generate_state_hash, initialise_env, initialise_state

logger = logger.get_logger(__name__)

@debug
def load_agent(config, algo=None) -> dict:
    """
        Loads pickle file as a python dict from specified path in config.

        Args:
            config (dict) : Config settings and variables
            algo (str) : String value with either ql or sarsa to determine which agent to load.

        Returns:
            dict: A dictionary of the pickle filepath parsed in.
    """

    if algo == 'ql':
        with open(config['predict']['best_ql_agent_filename'], 'rb') as handle:
            q = pickle.load(handle)
            handle.close()
        logger.info("Best Q-Learning Agent loaded.")
    elif algo == 'sarsa':
        with open(config['predict']['best_sarsa_agent_filename'], 'rb') as handle:
            q = pickle.load(handle)
            handle.close()
        logger.info("Best Sarsa Agent loaded.")
    else:
        with open(config['predict']['best_agent_filename'], 'rb') as handle:
            q = pickle.load(handle)
            handle.close()
        logger.info("Best Agent loaded.")
    return q

@debug
def plot_reward(config, rewards: List[float]) -> None:
    """
        Plots the cumulative sum of the rewards divided by the number of episodes and saves image to location specified in config.

        Args:
            config (dict) : Config settings and variables
            rewards (list) : Rewards accumulated at each episode
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
    plt.savefig(f"{config['predict']['pred_art_dir']}/reward_{int(time.time())}.png")
    logger.info(f"Rewards plot saved successfully.")
    plt.close(plt.gcf())

@debug
def predict(config, q, render=False) -> (Dict[str, Union[float, List]], List[float], List[float]):
    """
    Uses a pre-trained agent (q table) to predict actions for agent in an environment.

    Args:
        config (dict) : Config settings and variables
        q (dict): Pre-trained q table for all states
        render (default=False): Used to choose if displaying the agent in gridworld is required. Set to True if you would like to see the render.

    Returns:
        result: dict containing the avg_reward, completion_rate and avg_steps for the current agent over specified episodes.
        rewards: list containing reward achieved by agent per episode
        step_tracker: list containing number of steps taken by agent in each episode to reach goal.
    """
    env = initialise_env(config)
    state = initialise_state(env)
    rewards = []
    step_tracker = []
    epsilon = 0
    for episode in range(config['predict']['episodes']):
        state = initialise_state(env)
        for step in range(config['general']['max_steps']):
            action = np.argmax(q[state])
            _next_state, reward, done, info = env.step(action)
            if render:
                env.render() # render the environment, this does not work inside Jupyter notebook
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
            state = next_state
        rewards.append(reward)
    result = {'avg_reward': np.sum(rewards) / np.count_nonzero(rewards),
              'completion_rate': np.count_nonzero(rewards) / len(rewards),
              'avg_steps': np.sum(step_tracker) / len(step_tracker)}
    logger.info(f"Final output for agent: {result}")
    plot_reward(config, rewards)
    return result, rewards, step_tracker


@timer
@debug
def main(config_path: str, args) -> None:
    config = read_yaml(config_path)
    q = load_agent(config, args.algo)
    result, rewards, step_tracker = predict(config, q)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', metavar='a', type=str, help='Choose between ql and sarsa')
    args = parser.parse_args()

    config_path = "../conf/config.yaml"
    main(config_path, args)
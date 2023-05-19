from env.env import Env
from agent.QLearner import QLearnig, QLearningTest
from utils.agent import epsilon_greedy
import numpy as np
import gymnasium as gym

def taxi_q_learning():
    taxi_env = Env(env_name="Taxi-v3").create_env()
    agent = QLearnig(env=taxi_env)
    agent.init_q_table()
    agent.learn(
        episodes=2000,
        alpha=0.2,
        gamma=0.99
    )
    agent.plot_rewards()
    taxi_env.close()
    return agent

def taxi_test(agent: QLearnig, episodes:int):
    taxi_test_env = Env(env_name="Taxi-v3", render_mode="human").create_env()
    test = QLearningTest(taxi_test_env, agent)
    test.run_test(episodes)
    taxi_test_env.close()

def frozen_lake_q_learing():
    fl_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    agent = QLearnig(env=fl_env)
    agent.init_q_table()
    agent.learn(
        episodes=100000,
        alpha=0.2,
        gamma=0.99
    )
    agent.plot_rewards()
    fl_env.close()
    return agent

def frozen_lake_test(agent: QLearnig, episodes:int):
    fl_test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
    test = QLearningTest(fl_test_env, agent)
    test.run_test(episodes)
    fl_test_env.close()


if __name__ == "__main__":
    # agent = taxi_q_learning()
    # taxi_test(agent, 10)
    agent = frozen_lake_q_learing()
    frozen_lake_test(agent, 5)

    
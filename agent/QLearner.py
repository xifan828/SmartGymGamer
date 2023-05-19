import gymnasium as gym
import numpy as np
import time
from utils.agent import q_update_bellmann, epsilon_greedy, episodes_rewards_plot, select_epsilon

class QLearnig():
    def __init__(self, env:gym.Env):
        self.env = env
        self.observations = env.observation_space.n
        self.actions = env.action_space.n
    
    def init_q_table(self):
        self.q_table = np.zeros((self.observations, self.actions))
        return True
    
    def load_q_table(self, path:str):
        self.q_table = np.load(path)
        return True

    def learn(self, episodes: int, alpha, gamma):
        self.rewards_list = []
        for episode in range(episodes):
            print(f"episode {episode}")
            observation, _ = self.env.reset()
            terminated = False
            steps = 0
            rewards = 0
            while not terminated and steps < 50:
                epsilon = select_epsilon(episode)
                action = epsilon_greedy(
                    self.q_table,self.env,observation,epsilon
                )
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.q_table[observation, action] = q_update_bellmann(
                    self.q_table, observation, action, reward, next_observation, alpha, gamma
                    )
                observation = next_observation
                rewards += reward
                steps += 1
            self.rewards_list.append(rewards)
        self.episodes_list = list(range(1, episodes+1))
        print("training completed")
        return None
    
    def plot_rewards(self):
        fig = episodes_rewards_plot(self.episodes_list, self.rewards_list)

    def make_action(self, observation):
        return np.argmax(self.q_table[observation])

class QLearningTest():
    def __init__(self, env:gym.Env, agent:QLearnig):
        self.env = env
        self.agent = agent
    
    def run_test(self, episodes):
        for i in range(episodes):
            observation, _ = self.env.reset()
            terminated = False
            steps = 0
            while not terminated and steps < 50:
                action = self.agent.make_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                steps += 1




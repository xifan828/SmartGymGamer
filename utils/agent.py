import numpy as np
import plotly.express as px
import pandas as pd

def select_epsilon(episode):
    epsilon = max(0.01, 1 * (0.99 ** episode))
    return epsilon

def epsilon_greedy(q_table, env, observation, epsilon):
    random_number = np.random.uniform(0, 1)
    if random_number < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[observation])
    return action

def q_update_bellmann(q_table: np.array, observation: int, action:int, reward: int, next_observation:int, alpha:float, gamma: float):
    value = q_table[observation, action] + alpha * (reward + gamma * np.max(q_table[next_observation]) - q_table[observation, action])
    return value

def episodes_rewards_plot(episodes_list, rewards_list):
    results = pd.DataFrame(
        data={"episode": episodes_list, "rewards": rewards_list}
    )
    results["rewards_average"] = results["rewards"].rolling(window=20).mean()
    fig = px.line(
        results,
        x="episode",
        y="rewards_average",
        labels={"x": "episodes", "y": "rewards"},
        title="reward per episode"
    )
    fig.show()
    return fig 

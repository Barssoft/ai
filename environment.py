import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class HigherLowerEnv(gym.Env):
    def __init__(self):
        super(HigherLowerEnv, self).__init__()
        self.current_number = np.random.randint(0, 101)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.balance = 50
        self.history = []
        self.correct_predictions = 0
        self.total_predictions = 0

    def reset(self, seed=None, options=None):
        self.current_number = np.random.randint(0, 101)
        self.balance = 50
        self.history = [(self.current_number, None, self.balance)]
        self.correct_predictions = 0
        self.total_predictions = 0
        return np.array([self.current_number], dtype=np.float32), {}

    def step(self, action):
        next_number = np.random.randint(0, 101)
        self.total_predictions += 1
        reward = 2 if (action == 1 and next_number > self.current_number) or (action == 0 and next_number < self.current_number) else -3
        if reward == 2: self.correct_predictions += 1
        self.balance += reward
        done = self.balance <= 0
        self.history.append((next_number, action, self.balance))
        self.current_number = next_number
        return np.array([self.current_number], dtype=np.float32), reward, done, False, {}

    def render(self):
        plt.ion()  # Включаем интерактивный режим (чтобы график обновлялся, а не создавался заново)

        x = np.arange(len(self.history))
        y, actions, balances = zip(*self.history)

        plt.clf()  # Очищаем предыдущий график
        plt.plot(x, y, marker='o', label="Числа", linestyle="-")
        plt.plot(x, balances, marker='s', linestyle="--", color="blue", label="Баланс")

        for i, (num, action, bal) in enumerate(self.history):
            if action is not None:
                color = "green" if action == 1 else "red"
                plt.scatter(i, num, c=color, marker="x", s=100)

        plt.title(f"Баланс: {self.balance}")
        plt.legend()

        plt.draw()  # Перерисовываем график без закрытия окна
        plt.pause(0.1)  # Делаем небольшую паузу для обновления



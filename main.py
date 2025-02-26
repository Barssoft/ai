import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO


class HigherLowerEnv(gym.Env):
    def __init__(self):
        super(HigherLowerEnv, self).__init__()

        self.current_number = np.random.randint(0, 101)  # Первое случайное число
        self.action_space = spaces.Discrete(2)  # 0 = меньше, 1 = больше
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        self.balance = 50  # Стартовый баланс
        self.history = []  # История для графика
        self.correct_predictions = 0  # Подсчет успешных предсказаний
        self.total_predictions = 0  # Общее число попыток

    def reset(self, seed=None, options=None):
        self.current_number = np.random.randint(0, 101)
        self.balance = 50  # Обновляем баланс
        self.history = [(self.current_number, None, self.balance)]  # (число, действие, баланс)
        self.correct_predictions = 0  # Сбрасываем счетчик успешных предсказаний
        self.total_predictions = 0  # Сбрасываем общее число предсказаний
        return np.array([self.current_number], dtype=np.float32), {}

    def step(self, action):
        next_number = np.random.randint(0, 101)  # Генерируем следующее число
        self.total_predictions += 1  # Увеличиваем счетчик всех предсказаний

        # Проверяем, угадал ли агент
        if (action == 1 and next_number > self.current_number) or (action == 0 and next_number < self.current_number):
            reward = 2  # Угадал
            self.correct_predictions += 1  # Увеличиваем счетчик успешных предсказаний
        else:
            reward = -3  # Ошибся

        self.balance += reward  # Обновляем баланс
        done = self.balance <= 0  # Игра заканчивается, если баланс ≤ 0
        self.history.append((next_number, action, self.balance))  # Запоминаем данные

        self.current_number = next_number
        return np.array([self.current_number], dtype=np.float32), reward, done, False, {}

    def render(self):
        x = np.arange(len(self.history))
        y, actions, balances = zip(*self.history)

        plt.clf()
        plt.plot(x, y, marker='o', label="Числа", linestyle="-")
        plt.plot(x, balances, marker='s', linestyle="--", color="blue", label="Баланс")  # Баланс

        # Отмечаем предсказания
        for i, (num, action, bal) in enumerate(self.history):
            if action is not None:
                color = "green" if action == 1 else "red"
                plt.scatter(i, num, c=color, marker="x", s=100)

        plt.title(f"Баланс: {self.balance}")
        plt.legend()
        plt.pause(0.1)


# === Обучение агента ===
env = HigherLowerEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# === Тестируем обученную модель ===
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs)  # Предсказание нейросети
    obs, reward, done, _, _ = env.step(action)  # Выполняем шаг

    env.render()  # Визуализация
    time.sleep(0.1)

    if done:  # Если баланс 0, заканчиваем
        print("Игра окончена! Баланс закончился.")
        break

# === Вычисляем и выводим винрейт ===
winrate = (env.correct_predictions / env.total_predictions) * 100
print(f"Винрейт модели: {winrate:.2f}%")

plt.show()

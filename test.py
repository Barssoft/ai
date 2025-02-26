from stable_baselines3 import PPO
from environment import HigherLowerEnv

env = HigherLowerEnv()
model = PPO.load("higher_lower_model")

obs, _ = env.reset()
for _ in range(150):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done: break

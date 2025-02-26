from stable_baselines3 import PPO
from environment import HigherLowerEnv

env = HigherLowerEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("higher_lower_model")

import pickle
import time

import gym

env = gym.make('CartPole-v0')
t = 0
done = False
agent = pickle.load(open("../pickles/pytorch_agent_episode_10000.p", "rb"))

current_state = env.reset()
while not done:
    env.render()
    time.sleep(0.05)
    t = t + 1
    action, _ = agent.act(current_state)
    obs, reward, done, _ = env.step(action)
    current_state = obs

env.close()

print(f"finished after {t} timesteps")

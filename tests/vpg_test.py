import pickle

vpg_agent = pickle.load(open("../pickles/vpg_agent_episode_29000.p", "rb"))
action, action_prob = vpg_agent.act([0, 0, -0.17, 0])
print("oh god, help me")

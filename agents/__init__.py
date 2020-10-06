from abc import abstractmethod


class Agent:
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def store_transition(self, state, new_state, action, a_prob, reward):
        pass

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def save(self, iteration):
        pass

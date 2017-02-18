import threading

class RNNCriticTrainer(object):

    def __init__(self, policy):
        self._policy = policy
        self._is_async



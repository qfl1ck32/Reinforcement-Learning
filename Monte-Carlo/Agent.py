from collections import defaultdict
from gym import Env
from numpy import argmax, array, ones, arange, zeros
from numpy.random import rand, choice


def generate_probability():
    return rand()


def get_best_policy(q_values: dict):
    return dict((key, argmax(value)) for key, value in q_values.items())


class Agent:
    env: Env

    gamma: float  # discount
    alpha: float  # similar to "learning rate" in ML

    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    number_of_actions: int

    Q: defaultdict  # map [state][action] -> expected reward

    policy: defaultdict  # map [state] -> action

    def __init__(self, env: Env,
                 alpha=0.001,
                 gamma=1.0,
                 epsilon_start=1.0,
                 epsilon_end=0.5,
                 epsilon_decay=0.9999):
        self.env = env

        self.gamma = gamma
        self.alpha = alpha

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy = defaultdict(lambda: 0)

        self.Q = defaultdict(lambda: zeros(self.number_of_actions))

        self.number_of_actions = env.action_space.n

    def play(self, random_policy=False):
        state = self.env.reset()

        while True:
            print(state)

            try:
                self.env.render()
            except NotImplementedError:
                pass

            action = self.get_action_space_sample() if random_policy else self.policy[state]

            state, reward, done, info = self.env.step(action)

            if done:
                print(state, reward, done, info)
                print(f'Reward: {reward}, so, player {"won" if reward > 0 else "lost"}.\n')
                break

    def get_action_space_sample(self):
        return self.env.action_space.sample()

    def generate_epsilon_greedy_policy(self, q_values_for_current_state: dict, epsilon: float):
        policy = ones(self.number_of_actions) * epsilon / self.number_of_actions

        best_action_for_current_state = argmax(q_values_for_current_state)

        policy[best_action_for_current_state] = 1 - epsilon + (epsilon / self.number_of_actions)

        return policy

    def generate_epsilon(self, current_episode: int):
        return max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** current_episode))

    def generate_episode(self, epsilon: float):
        episode = []

        state = self.env.reset()

        while True:
            policy = self.generate_epsilon_greedy_policy(self.Q[state], epsilon)

            action = choice(arange(self.number_of_actions), p=policy) \
                if state in self.Q else self.get_action_space_sample()

            next_state, reward, done, info = self.env.step(action)

            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def update_policy(self, epsilon: float = None):
        for state, actions in self.Q.items():
            if epsilon is None:
                self.policy[state] = argmax(actions)
                continue

            if generate_probability() < epsilon:
                self.policy[state] = self.get_action_space_sample()
            else:
                self.policy[state] = argmax(actions)

    def update_q_table(self, episode: [any]):
        states, actions, rewards = zip(*episode)

        discounts = array([self.gamma ** i for i in range(len(rewards) + 1)])

        for index, state in enumerate(states):
            action = actions[index]

            old_q_value = self.Q[state][action]

            self.Q[state][action] = (1 - self.alpha) * old_q_value + self.alpha * \
                                    (sum(rewards[index:] * discounts[: -(index + 1)]))

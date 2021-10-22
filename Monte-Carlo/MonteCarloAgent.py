from Agent import Agent


class MonteCarloAgent(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def control(self, number_of_episodes: int):
        for current_episode in range(number_of_episodes):
            epsilon = self.generate_epsilon(current_episode)

            episode = self.generate_episode(epsilon)

            self.update_q_table(episode)

            self.update_policy()

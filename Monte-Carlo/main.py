import gym

from MonteCarloAgent import MonteCarloAgent


def main():

    agent = MonteCarloAgent(env=gym.make("Blackjack-v1"))

    agent.control(number_of_episodes=10000)

    print('''Done controlling.
Press any key to run an episode.
Write "exit" to leave.
''')

    while input() != 'exit':
        agent.play()


if __name__ == '__main__':
    main()

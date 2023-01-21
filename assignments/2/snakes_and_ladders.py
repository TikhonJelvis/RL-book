Transition = {}

# Define the ladders and snakes
ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 51: 67, 72: 91, 80: 99}
snakes = {32: 10, 36: 6, 48: 26, 62: 18, 88: 24, 95: 56, 97: 78}

# Initialize the transition probabilities for each state


def initialize_transition():
    for s in range(100):
        next_states = {}
        for a in range(1, 7):
            s_prime = min(s + a, 100)
            if s_prime in ladders:
                s_prime = ladders[s_prime]
            elif s_prime in snakes:
                s_prime = snakes[s_prime]
            next_states[s_prime] = 1/6
        Transition[s] = next_states


if __name__ == '__main__':
    initialize_transition()

    # Print Transition for all entries of s = 1
    print(Transition[1])

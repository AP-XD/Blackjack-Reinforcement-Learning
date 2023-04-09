# Blackjack-Reinforcement-Learning

Our project involves using reinforcement learning to develop an optimal strategy for playing Blackjack against a dealer. The game is finite-state and the ace card presents a challenge as it can be worth either 1 or 11 depending on the situation, creating two sets of states. The game is also stochastic due to the random card drawing process, but reinforcement learning is well-suited for handling this. We plan to train our agent using Q-learning, Sarsa, and temporal difference, each with one million training iterations. Once trained, we will evaluate their performance by having them play against the dealer for 100,000 games and recording the results.

After train for 10000000 iterations using Q-Learning\
Our game bot fights against the dealer for 100000 rounds\
Win: 42.373 %\
Draw: 8.696 %\
Lose: 48.931000000000004 %

After train for 10000000 iterations using Sarsa\
Our game bot fights against the dealer for 100000 rounds\
Win: 42.28 %\
Draw: 7.409000000000001 %\
Lose: 50.31099999999999 %

After train for 10000000 iterations using Temporal Difference\
Our game bot fights against the dealer for 100000 rounds\
Win: 43.15 %\
Draw: 8.927 %\
Lose: 47.923 %

Ok, three methods give more or less the same performance, with temporal difference method gives slightly better result. Next we report their strategy maps (H = hit, S = stick).

                                            Q-learning
              Player (usable ace)                            Player (no usable ace)
          11 12 13 14 15 16 17 18 19 20 21               11 12 13 14 15 16 17 18 19 20 21
       1   H  S  S  H  S  S  S  S  S  S  S            1   H  H  S  H  H  H  S  S  S  S  S
       2   H  H  H  H  H  H  H  S  S  S  S            2   H  H  S  H  S  S  S  S  S  S  S
    D  3   H  H  H  H  H  H  H  H  S  S  S        D   3   H  H  H  S  S  S  S  S  S  S  S
    e  4   H  H  H  H  H  H  H  S  S  S  S        e   4   H  H  S  S  S  S  S  S  S  S  S
    a  5   H  H  H  H  H  H  H  H  S  S  S        a   5   H  H  S  S  S  S  S  S  S  S  S
    l  6   H  H  H  H  H  H  H  H  S  S  S        l   6   H  H  S  S  S  S  S  S  S  S  S
    e  7   H  H  H  H  H  H  H  S  S  S  S        e   7   H  H  H  H  H  H  S  S  S  S  S
    r  8   H  H  H  H  H  H  H  S  S  S  S        r   8   H  H  H  H  H  H  H  S  S  S  S
       9   H  H  H  H  H  H  H  S  S  S  S            9   H  H  H  H  H  H  S  S  S  S  S
      10   H  H  H  H  H  H  S  S  S  S  S           10   H  H  H  H  S  H  S  S  S  S  S

                                              Sarsa
              Player (usable ace)                            Player (no usable ace)
          11 12 13 14 15 16 17 18 19 20 21               11 12 13 14 15 16 17 18 19 20 21
       1   H  S  S  S  S  S  S  S  S  S  S            1   H  S  S  S  H  H  S  S  S  S  S
       2   H  H  H  H  H  H  S  S  S  S  S            2   H  S  S  S  S  S  S  S  S  S  S
    D  3   H  H  H  H  H  H  S  H  S  S  S        D   3   H  S  S  S  S  S  S  S  S  S  S
    e  4   H  H  H  H  H  H  H  S  S  S  S        e   4   H  S  S  S  S  S  S  S  S  S  S
    a  5   H  H  H  H  H  H  H  H  S  S  S        a   5   H  S  S  S  S  S  S  S  S  S  S
    l  6   H  H  H  H  H  H  H  H  S  S  S        l   6   H  S  S  S  S  S  S  S  S  S  S
    e  7   H  H  H  H  H  H  S  S  S  S  S        e   7   H  H  S  H  H  H  S  S  S  S  S
    r  8   H  H  H  H  H  H  H  S  S  S  S        r   8   H  H  H  H  H  H  H  S  S  S  S
       9   H  H  H  H  H  H  H  S  S  S  S            9   H  S  H  H  S  S  S  S  S  S  S
      10   H  H  H  S  S  S  S  S  S  S  S           10   H  S  S  S  H  H  S  S  S  S  S

                                        Temporal Difference
              Player (usable ace)                            Player (no usable ace)
          11 12 13 14 15 16 17 18 19 20 21               11 12 13 14 15 16 17 18 19 20 21
       1   H  H  H  H  H  H  H  H  S  S  S            1   H  H  H  H  H  H  S  S  S  S  S
       2   H  H  H  H  H  H  H  S  S  S  S            2   H  S  S  S  S  S  S  S  S  S  S
    D  3   H  H  H  H  H  H  H  S  S  S  S        D   3   H  S  S  S  S  S  S  S  S  S  S
    e  4   H  H  H  H  H  H  H  S  S  S  S        e   4   H  S  S  S  S  S  S  S  S  S  S
    a  5   H  H  H  H  H  H  H  S  S  S  S        a   5   H  S  S  S  S  S  S  S  S  S  S
    l  6   H  H  H  H  H  H  H  S  S  S  S        l   6   H  S  S  S  S  S  S  S  S  S  S
    e  7   H  H  H  H  H  H  S  S  S  S  S        e   7   H  H  H  H  H  S  S  S  S  S  S
    r  8   H  H  H  H  H  H  H  S  S  S  S        r   8   H  H  S  H  H  H  S  S  S  S  S
       9   H  H  H  H  H  H  H  H  S  S  S            9   H  H  H  H  H  H  S  S  S  S  S
      10   H  H  H  H  H  H  H  S  S  S  S           10   H  H  H  H  H  H  S  S  S  S  S


I have shown the difference between the three.

The code I've developed for this project allows for some flexibility in the gameplay. For instance, players can customize their own deck, set the winning score they desire, and choose the dealer's rules. All of these features are described in detail in the main.py file located within the project repository.

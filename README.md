# 2048
 C++ iterative implementation of 2048
This program uses a Beam Search algorithm with a static depth to find the best possible move.
It is a very simple algorithm with no ML, no intricate heuristics and no Pruning.
You can improve this algorithm hugely by removing the representation of the whole decision tree and instead use 2 arrays to store the current and previous layers, while also keeping in these arrays the paths you had to take to get there.
You can greatly improve this by adding a MCTS agent.
2048 environment and MCTS simulation
```
File contents:
├── decision_tree 
│   ├── decision_tree.cpp
│   ├── MCTS
│   ├── Py_Master
│   │   ├── constants.py
│   │   ├── logic.py
│   │   ├── puzzle.py
│   │   └── __pycache__
│   │       ├── constants.cpython-312.pyc
│   │       └── logic.cpython-312.pyc
│   └── tree.txt
├── Documentatie.pdf
├── LICENSE
└── README.md
```
How to use the MCTS inside GUI:
Technically, just run `puzzle.py` and it should work. 
If you want to change the structure go into `decision_tree.cpp`, change the absolute path of `tree.txt` to match your structure, build, go into `Py_Master -> puzzle.py` and change the paths of `tree.txt` and `MCTS.exe` to match your own.

Then run puzzle.py. Use the `arrow keys` to make your move the key `N` to get the best move (as computed by the MCTS). 

if you wish to use the MCTS without the GUI, you can simply edit tree.txt with the board you want it to evaluate and call `MCTS`
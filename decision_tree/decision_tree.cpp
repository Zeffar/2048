#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

using namespace std;

struct Node {
    int board[16]; // Flattened 4x4 board
    int parent_move;
    int score;
    int visits;
    int moves;
    long long total_score;

    // Constructor
    Node(int* a = nullptr, int move = -1) : parent_move(move), score(0), visits(0), moves(0), total_score(0) {
        if (a) memcpy(this->board, a, sizeof(this->board));
        else memset(this->board, 0, sizeof(this->board));
    }
};

void move_up(bool& ok, int* board);
void move_down(bool& ok, int* board);
void move_left(bool& ok, int* board);
void move_right(bool& ok, int* board);
void generate_2(int* board, mt19937& rng);
int simulate_random_play(Node& state, int& root_move, mt19937& rng);
int MCTS(Node& root, mt19937& rng);

int main() {
    ifstream read_tree("decision_tree/tree.txt");
    mt19937 rng(static_cast<unsigned>(time(nullptr))); // Efficient RNG seeded once

    Node root;
    for (int i = 0; i < 16; ++i) {
        int x;
        read_tree >> x;
        root.board[i] = x ? static_cast<int>(log2(x)) : 0;
    }

    // clock_t start = clock();
    cout << MCTS(root, rng) << endl;
    // clock_t end = clock();
    // double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    // cout << "Elapsed time: " << elapsed_secs << " seconds" << endl;
    return 0;
}

int MCTS(Node& root, mt19937& rng) {
    int zerocount = count(begin(root.board), end(root.board), 0);
    // int runs = zerocount > 7 ? 500 : (zerocount > 2 ? 1000 : 2000);
    int runs = 2000;

    long long scores[4] = {0, 0, 0, 0};
    long long visits[4] = {0, 0, 0, 0};

    for (int iterations = 0; iterations < runs; ++iterations) {
        int first_move = -1;
        int result = simulate_random_play(root, first_move, rng);
        if (first_move != -1) {
            scores[first_move] += result;
            visits[first_move]++;
        }
    }

    int best_score = -1, move = -1;
    for (int i = 0; i < 4; ++i) {
        if (visits[i] > 0) {
            long long avg_score = scores[i] / visits[i];
            if (avg_score > best_score) {
                best_score = avg_score;
                move = i;
            }
        }
    }
    return move;
}

int simulate_random_play(Node& state, int& root_move, mt19937& rng) {
    Node new_state(state.board); // Reuse node
    int moves = 0;
    int directions[4] = {0, 1, 2, 3};

    while (true) {
        bool game_over = true;
        shuffle(begin(directions), end(directions), rng);

        for (int move : directions) {
            bool legal_move = false;
            switch (move) {
                case 0: move_up(legal_move, new_state.board); break;
                case 1: move_right(legal_move, new_state.board); break;
                case 2: move_down(legal_move, new_state.board); break;
                case 3: move_left(legal_move, new_state.board); break;
            }

            if (legal_move) {
                generate_2(new_state.board, rng);
                if (root_move == -1) root_move = move;
                moves++;
                game_over = false;
                break;
            }
        }

        if (game_over) break;
    }
    return moves;
}

void generate_2(int* board, mt19937& rng) {
    vector<int> empty_tiles;
    for (int i = 0; i < 16; ++i)
        if (board[i] == 0)
            empty_tiles.push_back(i);

    if (!empty_tiles.empty()) {
        uniform_int_distribution<int> dist(0, empty_tiles.size() - 1);
        board[empty_tiles[dist(rng)]] = 1;
    }
}

void move_left(bool& ok, int* board) {
    for (int i = 0; i < 4; ++i) {
        int* row = board + i * 4;
        int write_idx = 0, prev = -1;
        for (int j = 0; j < 4; ++j) {
            if (row[j] != 0) {
                if (prev == row[j]) {
                    row[write_idx - 1]++; // Merge tiles
                    prev = -1;
                    ok = true;
                } else {
                    if (j != write_idx) ok = true;
                    row[write_idx++] = row[j];
                    prev = row[j];
                }
            }
        }
        while (write_idx < 4) row[write_idx++] = 0;
    }
}

void move_right(bool& ok, int* board) {
    for (int i = 0; i < 4; ++i) {
        int* row = board + i * 4;
        int write_idx = 3, prev = -1;
        for (int j = 3; j >= 0; --j) {
            if (row[j] != 0) {
                if (prev == row[j]) {
                    row[write_idx + 1]++; // Merge tiles
                    prev = -1;
                    ok = true;
                } else {
                    if (j != write_idx) ok = true;
                    row[write_idx--] = row[j];
                    prev = row[j];
                }
            }
        }
        while (write_idx >= 0) row[write_idx--] = 0;
    }
}

void move_up(bool& ok, int* board) {
    for (int col = 0; col < 4; ++col) {
        int write_idx = 0, prev = -1;
        for (int i = 0; i < 4; ++i) {
            int& cell = board[i * 4 + col];
            if (cell != 0) {
                if (prev == cell) {
                    board[(write_idx - 1) * 4 + col]++; // Merge tiles
                    prev = -1;
                    ok = true;
                } else {
                    if (i != write_idx) ok = true;
                    board[write_idx * 4 + col] = cell;
                    write_idx++;
                    prev = cell;
                }
            }
        }
        while (write_idx < 4) board[write_idx++ * 4 + col] = 0;
    }
}

void move_down(bool& ok, int* board) {
    for (int col = 0; col < 4; ++col) {
        int write_idx = 3, prev = -1;
        for (int i = 3; i >= 0; --i) {
            int& cell = board[i * 4 + col];
            if (cell != 0) {
                if (prev == cell) {
                    board[(write_idx + 1) * 4 + col]++; // Merge tiles
                    prev = -1;
                    ok = true;
                } else {
                    if (i != write_idx) ok = true;
                    board[write_idx * 4 + col] = cell;
                    write_idx--;
                    prev = cell;
                }
            }
        }
        while (write_idx >= 0) board[write_idx-- * 4 + col] = 0;
    }
}

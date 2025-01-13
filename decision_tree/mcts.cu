#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h> // For random number generation on the GPU
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

#define THREADS_PER_BLOCK 256

using namespace std;

struct Node {
    int board[16]; 
    int parent_move;
    int score;
    int visits;
    int moves;
    long long total_score;

    Node(int* a = nullptr, int move = -1) : parent_move(move), score(0), visits(0), moves(0), total_score(0) {
        if (a) memcpy(this->board, a, sizeof(this->board));
        else memset(this->board, 0, sizeof(this->board));
    }
};

// CUDA kernel for initializing RNG states
__global__ void init_rng(curandState* states, int num_simulations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_simulations) {
        curand_init(1234, idx, 0, &states[idx]);
    }
}

__device__ void move_up(bool& ok, int* board);
__device__ void move_down(bool& ok, int* board);
__device__ void move_left(bool& ok, int* board);
__device__ void move_right(bool& ok, int* board);
__device__ void generate_2(int* board, curandState* state);
__global__ void simulate_random_play_kernel(int* root_board, int num_simulations, int* results, curandState* rng_states);
void simulate_random_play_cuda(int* root_board, int num_simulations, int* results);
// int simulate_random_play(Node& state, int& root_move, mt19937& rng);
int MCTS(Node& root, int num_simulations);

int main() {
    ifstream read_tree("decision_tree/tree.txt");

    Node root;
    for (int i = 0; i < 16; ++i) {
        int x;
        read_tree >> x;
        root.board[i] = x ? static_cast<int>(log2(x)) : 0;
    }

    int num_simulations = 1;

    // auto start = chrono::high_resolution_clock::now();
    int best_move = MCTS(root, num_simulations);
    // auto end = chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed = end - start;

    // cout << "Best move: " << best_move << endl;
    // cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;
    cout << best_move << endl;

    return 0;
}

int MCTS(Node& root, int num_simulations) {
    vector<int> results(num_simulations);

    // Call CUDA function
    simulate_random_play_cuda(root.board, num_simulations, results.data());

    long long scores[4] = {0, 0, 0, 0};
    long long visits[4] = {0, 0, 0, 0};

    // Aggregate results from multiple trials
    for (int trial = 0; trial < num_simulations; ++trial) {
        int first_move = trial % 4; // Example: Assign a move based on trial index
        scores[first_move] += results[trial];
        visits[first_move]++;
    }

    // Find the best move
    int best_score = -1, move = -1;
    for (int i = 0; i < 4; ++i) {
        // cout << "Move: " << i << ", Visits: " << visits[i] << ", Score: " << scores[i] << endl;
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


void simulate_random_play_cuda(int* root_board, int num_simulations, int* results) {
    int* d_root_board;
    int* d_results;
    curandState* d_rng_states;

    // Allocate device memory
    cudaMalloc(&d_root_board, sizeof(int) * 16);
    cudaMalloc(&d_results, sizeof(int) * num_simulations);
    cudaMalloc(&d_rng_states, sizeof(curandState) * THREADS_PER_BLOCK * num_simulations);

    // Copy the root board to the device
    cudaMemcpy(d_root_board, root_board, sizeof(int) * 16, cudaMemcpyHostToDevice);

    // Initialize RNG states
    int num_blocks = num_simulations;
    int shared_memory_size = THREADS_PER_BLOCK * sizeof(int); // Shared memory size per block
    init_rng<<<num_blocks, THREADS_PER_BLOCK>>>(d_rng_states, THREADS_PER_BLOCK * num_simulations);
    cudaDeviceSynchronize();

    cout << "Launching kernel with " << num_simulations << " blocks and " 
    << THREADS_PER_BLOCK << " threads per block." << std::endl;
    // Launch the kernel
    simulate_random_play_kernel<<<num_blocks, THREADS_PER_BLOCK, shared_memory_size>>>(
        d_root_board, num_simulations, d_results, d_rng_states
    );
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(results, d_results, sizeof(int) * num_simulations, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_root_board);
    cudaFree(d_results);
    cudaFree(d_rng_states);
}


__global__ void simulate_random_play_kernel(
    int* root_board, int num_simulations, int* results, curandState* rng_states) {
    
    extern __shared__ int shared_results[]; // Shared memory for reduction
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int idx = block_id * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        printf("Block %d executing %d threads\n", blockIdx.x, blockDim.x);
    }
    // printf("Block %d, Thread %d performing rollout\n", blockIdx.x, threadIdx.x);


    // printf("Thread %d executing\n", idx); // Debug output

    // Initialize RNG for this thread
    curandState local_state = rng_states[idx];
    
    // Copy the root board to thread-local memory
    int board[16];
    for (int i = 0; i < 16; ++i) board[i] = root_board[i];

    // Perform a rollout
    int moves = 0;
    int directions[4] = {0, 1, 2, 3};

    while (true) {
        bool game_over = true;

        // Shuffle directions
        for (int i = 4 - 1; i > 0; --i) {
            int j = curand(&local_state) % (i + 1);
            int tmp = directions[i];
            directions[i] = directions[j];
            directions[j] = tmp;
        }

        // Attempt moves
        for (int move : directions) {
            bool legal_move = false;
            switch (move) {
                case 0: move_up(legal_move, board); break;
                case 1: move_right(legal_move, board); break;
                case 2: move_down(legal_move, board); break;
                case 3: move_left(legal_move, board); break;
            }

            if (legal_move) {
                generate_2(board, &local_state);
                moves++;
                game_over = false;
                break;
            }
        }

        if (game_over) break;
    }

    // Save the number of moves this thread performed
    shared_results[thread_id] = moves;
    __syncthreads();

    // Reduction within the block to compute the total moves
    if (thread_id == 0) {
        int total_moves = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            total_moves += shared_results[i];
        }
        results[block_id] = total_moves / blockDim.x; // Average moves for this trial
    }
}

__device__ void generate_2(int* board, curandState* state) {
    int empty_tiles[16];
    int num_empty = 0;

    for (int i = 0; i < 16; ++i) {
        if (board[i] == 0) {
            empty_tiles[num_empty++] = i;
        }
    }

    if (num_empty > 0) {
        int idx = curand(state) % num_empty;
        board[empty_tiles[idx]] = 1;
    }
}

__device__ void move_left(bool& ok, int* board) {
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

__device__ void move_right(bool& ok, int* board) {
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

__device__ void move_up(bool& ok, int* board) {
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

__device__ void move_down(bool& ok, int* board) {
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

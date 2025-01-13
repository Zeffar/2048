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
    int directions[MAX_MOVES] = {0, 1, 2, 3};

    while (true) {
        bool game_over = true;

        // Shuffle directions
        for (int i = MAX_MOVES - 1; i > 0; --i) {
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
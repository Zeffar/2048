import math
import random
import cupy as cp
import numpy as np
from graphviz import Digraph

# ----------------------------
#  Tree Visualization
# ----------------------------

def visualize_tree(root, max_nodes=4000):
    dot = Digraph(comment="MCTS Tree")
    def make_label(node):
        label = f"Visits: {node.visits}\nValue: {node.total_value:.2f}"
        if node.move_from_parent is not None:
            label += f"\nMove: {node.move_from_parent}"
        return label

    node_id_map = {}
    node_id_map[root] = 0
    dot.node("0", label=make_label(root))

    current_layer = [root]
    layer = 0

    while current_layer and len(node_id_map) < max_nodes:
        next_layer_candidates = []
        for node in current_layer:
            for child in node.children:
                if len(node_id_map) >= max_nodes:
                    break
                if child not in node_id_map:
                    if child.visits > 0:
                        metric = child.total_value / child.visits
                    else:
                        metric = 0.0  

                    parent_id = node_id_map[node]
                    next_layer_candidates.append((child, parent_id, metric))
        
        if not next_layer_candidates:
            break

        next_layer_candidates.sort(key=lambda x: x[2], reverse=True)
        next_layer_candidates = next_layer_candidates[:16]

        next_layer = []
        for child, parent_id, metric in next_layer_candidates:
            if child not in node_id_map:
                new_id = len(node_id_map)
                node_id_map[child] = new_id
                dot.node(str(new_id), label=make_label(child))
                dot.edge(str(parent_id), str(new_id))
                next_layer.append(child)
            else:
                child_id = node_id_map[child]
                dot.edge(str(parent_id), str(child_id))
        
        current_layer = next_layer
        layer += 1

    return dot

# ----------------------------
#  2048 CPU Logic (merges)
# ----------------------------


def _compress_merge_left(row):
    """Merge one row to the left (standard 2048)."""
    filtered = [val for val in row if val != 0]
    merged = []
    skip = False
    for i in range(len(filtered)):
        if skip:
            skip = False
            continue
        if i < len(filtered) - 1 and filtered[i] == filtered[i + 1]:
            merged.append(filtered[i] * 2)
            skip = True
        else:
            merged.append(filtered[i])
    merged += [0] * (4 - len(merged))
    return merged

def _move_left(board4x4):
    """Apply 'move left' to the board."""
    new_board = []
    changed = False
    for r in range(4):
        row = board4x4[r*4 : r*4+4]
        merged = _compress_merge_left(row)
        if merged != row:
            changed = True
        new_board.extend(merged)
    return new_board, changed

def _move_right(board4x4):
    """Apply 'move right' to the board."""
    new_board = []
    changed = False
    for r in range(4):
        row = board4x4[r*4 : r*4+4]
        row.reverse()
        merged = _compress_merge_left(row)
        merged.reverse()
        if merged != board4x4[r*4 : r*4+4]:
            changed = True
        new_board.extend(merged)
    return new_board, changed

def _transpose(board4x4):
    trans = [0]*16
    for r in range(4):
        for c in range(4):
            trans[c*4 + r] = board4x4[r*4 + c]
    return trans

def _move_up(board4x4):
    """Apply 'move up' by transposing -> move_left -> transpose back."""
    t = _transpose(board4x4)
    moved, changed = _move_left(t)
    moved = _transpose(moved)
    return moved, changed

def _move_down(board4x4):
    """Apply 'move down' by transposing -> move_right -> transpose back."""
    t = _transpose(board4x4)
    moved, changed = _move_right(t)
    moved = _transpose(moved)
    return moved, changed

def spawn_random_tile(board4x4):
    """Spawn a 2 (90%) or 4 (10%) in a random empty cell."""
    empties = [i for i, v in enumerate(board4x4) if v == 0]
    if not empties:
        return
    idx = random.choice(empties)
    board4x4[idx] = 2 if random.random() < 0.9 else 4

def simulate_move_cpu(board4x4, move_id):
    """
    If the move changes the board, spawn a tile and return the new board.
    Otherwise, return None.
    """
    new_b = board4x4[:]
    if move_id == 0:
        moved, changed = _move_up(new_b)
    elif move_id == 1:
        moved, changed = _move_right(new_b)
    elif move_id == 2:
        moved, changed = _move_down(new_b)
    else:
        moved, changed = _move_left(new_b)
    if not changed:
        return None
    spawn_random_tile(moved)
    return moved

def get_legal_moves(board4x4):
    """Return a list of valid moves (0..3) that actually change the board."""
    moves = []
    for m in [0,1,2,3]:
        possible = simulate_move_cpu(board4x4, m)
        if possible is not None:
            moves.append(m)
    return moves

def is_game_over_cpu(board4x4):
    """Check if no moves remain."""
    for m in [0,1,2,3]:
        sim = simulate_move_cpu(board4x4, m)
        if sim is not None:
            return False
    return True

# ----------------------------
#  GPU random rollout kernel
# ----------------------------

kernel_code = r'''
extern "C" __global__
void random_rollout_kernel(const int* boards,
                           const int n,
                           const int n_rollouts,
                           int* results,
                           long long* rng_states)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = tid; i < n; i += stride)
    {
        int loc[16];
        for(int k=0; k<16; k++){
            loc[k] = boards[i*16 + k];
        }
        long long rng = rng_states[tid];
        int moveCount = 0;

        for(int step=0; step < n_rollouts; step++)
        {
            bool can_move = false;
            for(int x=0; x<16; x++){
                if(loc[x] == 0){
                    can_move = true;
                    break;
                }
            }
            if(!can_move){
                for(int row=0; row<4 && !can_move; row++){
                    for(int col=0; col<3; col++){
                        int idx = row*4 + col;
                        if(loc[idx] != 0 && loc[idx] == loc[idx+1]){
                            can_move = true;
                            break;
                        }
                    }
                }
                for(int col=0; col<4 && !can_move; col++){
                    for(int row=0; row<3; row++){
                        int idx1 = row*4 + col;
                        int idx2 = (row+1)*4 + col;
                        if(loc[idx1] != 0 && loc[idx1] == loc[idx2]){
                            can_move = true;
                            break;
                        }
                    }
                }
            }
            if(!can_move) {
                break;
            }

            int r = (int)(rng & 3);
            rng = (rng ^ 0x9e3779b97f4a7c15ULL) * 6364136223846793005ULL + 1ULL;
            bool changed = false;

            auto moveLeft = [&](int* board){
                for(int row=0; row<4; row++){
                    int tmp[4]; 
                    int idx=0;
                    for(int c=0; c<4; c++){
                        int val = board[row*4 + c];
                        if(val != 0){
                            tmp[idx++] = val;
                        }
                    }
                    int merged[4];
                    int w=0;
                    for(int read=0; read<idx; read++){
                        if(read<idx-1 && tmp[read] == tmp[read+1]){
                            merged[w++] = tmp[read]*2;
                            read++;
                        } else {
                            merged[w++] = tmp[read];
                        }
                    }
                    while(w<4) merged[w++] = 0;
                    for(int c=0; c<4; c++){
                        int oldv = board[row*4 + c];
                        int newv = merged[c];
                        if(oldv != newv) changed = true;
                        board[row*4 + c] = newv;
                    }
                }
            };

            auto moveRight = [&](int* board){
                for(int row=0; row<4; row++){
                    int rev[4];
                    for(int c=0; c<4; c++){
                        rev[c] = board[row*4 + (3-c)];
                    }
                    int tmp[4];
                    int idx=0;
                    for(int c=0; c<4; c++){
                        if(rev[c] != 0){
                            tmp[idx++] = rev[c];
                        }
                    }
                    int merged[4];
                    int w=0;
                    for(int read=0; read<idx; read++){
                        if(read<idx-1 && tmp[read] == tmp[read+1]){
                            merged[w++] = tmp[read]*2;
                            read++;
                        } else {
                            merged[w++] = tmp[read];
                        }
                    }
                    while(w<4) merged[w++] = 0;
                    for(int c=0; c<4; c++){
                        int newv = merged[3-c];
                        int oldv = board[row*4 + c];
                        if(oldv != newv) changed = true;
                        board[row*4 + c] = newv;
                    }
                }
            };

            auto transpose = [&](int* arr){
                int tmp[16];
                for(int rr=0; rr<4; rr++){
                    for(int cc=0; cc<4; cc++){
                        tmp[cc*4 + rr] = arr[rr*4 + cc];
                    }
                }
                for(int z=0; z<16; z++){
                    arr[z] = tmp[z];
                }
            };

            auto moveUp = [&](int* board){
                int temp[16];
                for(int x=0; x<16; x++) temp[x] = board[x];
                transpose(temp);
                for(int row=0; row<4; row++){
                    int rowData[4];
                    for(int c=0; c<4; c++){
                        rowData[c] = temp[row*4 + c];
                    }
                    int idx=0;
                    int nonz[4];
                    for(int c=0; c<4; c++){
                        if(rowData[c]!=0) {
                            nonz[idx++] = rowData[c];
                        }
                    }
                    int merged[4];
                    int w=0;
                    for(int rd=0; rd<idx; rd++){
                        if(rd<idx-1 && nonz[rd]==nonz[rd+1]){
                            merged[w++] = nonz[rd]*2;
                            rd++;
                        } else {
                            merged[w++] = nonz[rd];
                        }
                    }
                    while(w<4) merged[w++] = 0;
                    for(int c=0; c<4; c++){
                        if(temp[row*4 + c] != merged[c]) changed=true;
                        temp[row*4 + c] = merged[c];
                    }
                }
                transpose(temp);
                for(int m=0; m<16; m++){
                    if(board[m]!=temp[m]) changed=true;
                    board[m] = temp[m];
                }
            };

            auto moveDown = [&](int* board){
                int temp[16];
                for(int x=0; x<16; x++) temp[x]=board[x];
                transpose(temp);
                for(int row=0; row<4; row++){
                    int rev[4];
                    for(int c=0; c<4; c++){
                        rev[c] = temp[row*4 + (3-c)];
                    }
                    int tmp_[4];
                    int idx=0;
                    for(int c=0; c<4; c++){
                        if(rev[c]!=0) tmp_[idx++] = rev[c];
                    }
                    int merged[4];
                    int w=0;
                    for(int rd=0; rd<idx; rd++){
                        if(rd<idx-1 && tmp_[rd]==tmp_[rd+1]){
                            merged[w++] = tmp_[rd]*2;
                            rd++;
                        } else {
                            merged[w++] = tmp_[rd];
                        }
                    }
                    while(w<4) merged[w++] = 0;
                    for(int c=0; c<4; c++){
                        int newv = merged[3-c];
                        if(temp[row*4 + c]!=newv) changed=true;
                        temp[row*4 + c] = newv;
                    }
                }
                transpose(temp);
                for(int m=0; m<16; m++){
                    if(board[m]!=temp[m]) changed=true;
                    board[m]=temp[m];
                }
            };

            if(r==0) moveUp(loc);
            else if(r==1) moveRight(loc);
            else if(r==2) moveDown(loc);
            else moveLeft(loc);

            if(changed){
                int empties[16];
                int ec=0;
                for(int x=0; x<16; x++){
                    if(loc[x]==0){
                        empties[ec++] = x;
                    }
                }
                if(ec>0){
                    int idxEmpty = (rng & 0xffff) % ec;
                    rng = (rng ^ 0x9e3779b97f4a7c15ULL) * 6364136223846793005ULL + 1ULL;
                    int spawnPos = empties[idxEmpty];

                    int r2 = (int)(rng & 0xffff);
                    rng = (rng ^ 0x9e3779b97f4a7c15ULL) * 6364136223846793005ULL + 1ULL;
                    if(r2 < 58982){
                        loc[spawnPos] = 2;
                    } else {
                        loc[spawnPos] = 4;
                    }
                }
                moveCount++;
            }
        }
        results[i] = moveCount;
        rng_states[tid] = rng;
    }
}
'''.strip()

random_rollout_kernel = cp.RawKernel(kernel_code, "random_rollout_kernel")

def run_random_rollouts(boards, n_rollouts=50):
    """
    boards: CuPy array (N,16)
    returns: CuPy array (N,) with final move counts
    """
    N = boards.shape[0]
    results = cp.zeros((N,), dtype=cp.int32)
    rng_states = cp.random.randint(0, 2**31-1, size=N, dtype=cp.int64)

    threads = 128
    blocks = (N + threads - 1) // threads
    random_rollout_kernel((blocks,), (threads,),
        (boards, N, n_rollouts, results, rng_states)
    )
    return results

# ----------------------------
#  MCTS on CPU
# ----------------------------

class MCTSNodeCPU:
    def __init__(self, board, parent=None, move_from_parent=None):
        self.board = board[:]
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.untried_moves = get_legal_moves(self.board)
        self.children = []
        self.visits = 0
        self.total_value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.41):
        best_node = None
        best_ucb = float('-inf')
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                avg_score = (child.total_value / child.visits)
                explore = c * math.sqrt((2.0 * math.log(self.visits)) / child.visits)
                ucb = avg_score + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child
        return best_node

    def add_child(self, move_id, new_board):
        child = MCTSNodeCPU(new_board, parent=self, move_from_parent=move_id)
        self.children.append(child)
        self.untried_moves.remove(move_id)
        return child

    def update(self, value):
        self.visits += 1
        self.total_value += value

def expand_cpu(node):
    if not node.untried_moves:
        return node
    m = random.choice(node.untried_moves)
    new_b = simulate_move_cpu(node.board, m)
    if new_b is None:
        node.untried_moves.remove(m)
        return node
    return node.add_child(m, new_b)

def tree_policy_cpu(node, c=1.41):
    for _ in range(100):
        if is_game_over_cpu(node.board):
            return node
        if not node.is_fully_expanded():
            return expand_cpu(node)
        else:
            node = node.best_child(c)
            if node is None:
                return node
    return node

def backup_cpu(node, value):
    while node is not None:
        node.update(value)
        node = node.parent

def default_policy_gpu(board, rollouts=64):
    boards_cpu = np.tile(board, (rollouts,1))
    boards_gpu = cp.asarray(boards_cpu, dtype=cp.int32)
    scores = run_random_rollouts(boards_gpu, n_rollouts=50)
    return float(cp.mean(scores))

def mcts_gpu(root_board, iterations=1000, c=1.41):
    root = MCTSNodeCPU(root_board)
    for _ in range(iterations):
        leaf = tree_policy_cpu(root, c)
        value = default_policy_gpu(leaf.board, rollouts=64)
        backup_cpu(leaf, value)

    best_move = None
    best_score = float('-inf')
    for child in root.children:
        if child.visits > 0:
            avg = child.total_value / child.visits
            if avg > best_score:
                best_score = avg
                best_move = child.move_from_parent

    tree_visualization = visualize_tree(root, max_nodes=2000)
    tree_visualization.render("decision_tree/mcts_cuda/mcts_tree.png", format="png", cleanup=True)  # Saves as 'mcts_tree.png'
    tree_visualization
    return best_move



# ----------------------------
#  Demo
# ----------------------------
if __name__ == "__main__":
    cp.random.seed(42)
    random.seed(42)
    board = np.loadtxt('decision_tree/mcts_cuda/state.txt', 
                       dtype=np.int32).flatten().tolist()
    print("Initial Board:")
    for r in range(4):
        print(board[r*4 : r*4+4])
    move_id = mcts_gpu(board, iterations=1000, c=2.0)
    moves = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT"}
    print(f"\nMCTS suggests move: {moves.get(move_id, 'None')}")


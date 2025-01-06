import math
import random
from graphviz import Digraph

def index_rc(r, c):
    """Returns the 1D index for (row, col) in a 4x4 board."""
    return (r << 2) + c

def spawn_tile_1d(board):
    """Spawn a 2 (exponent=1) or 4 (exponent=2) in a random empty cell."""
    empties = [i for i, v in enumerate(board) if v == 0]
    if not empties:
        return
    idx = random.choice(empties)
    board[idx] = 1 if random.random() < 0.9 else 2

def is_game_over_1d(board):
    """Check if no moves remain (no empty cells and no adjacent merges)."""
    if any(v == 0 for v in board):
        return False
    # Check rows
    for r in range(4):
        for c in range(3):
            idx1 = index_rc(r, c)
            idx2 = index_rc(r, c+1)
            if board[idx1] == board[idx2]:
                return False
    # Check columns
    for c in range(4):
        for r in range(3):
            idx1 = index_rc(r, c)
            idx2 = index_rc(r+1, c)
            if board[idx1] == board[idx2]:
                return False
    return True


ROW_MAP = {}  # (r0, r1, r2, r3) -> (merged_row, changed)

def _merge_left_4(row):
    arr = list(row)
    changed = False
    # Merge
    for i in range(3):
        if arr[i] != 0 and arr[i] == arr[i+1]:
            arr[i] += 1
            arr[i+1] = 0
            changed = True
    # Compact
    merged = [x for x in arr if x != 0]
    merged += [0]*(4 - len(merged))
    if tuple(merged) != row:
        changed = True
    return tuple(merged), changed

for r0 in range(16):
    for r1 in range(16):
        for r2 in range(16):
            for r3 in range(16):
                original = (r0, r1, r2, r3)
                ROW_MAP[original] = _merge_left_4(original)

def move_left_1d(board):
    """In-place 'move left' on the 1D board."""
    changed = False
    for r in range(4):
        row = (board[index_rc(r,0)], board[index_rc(r,1)],
               board[index_rc(r,2)], board[index_rc(r,3)])
        new_row, ch = ROW_MAP[row]
        if ch:
            changed = True
        board[index_rc(r,0)] = new_row[0]
        board[index_rc(r,1)] = new_row[1]
        board[index_rc(r,2)] = new_row[2]
        board[index_rc(r,3)] = new_row[3]
    return changed

def move_right_1d(board):
    """In-place 'move right' = reverse row, merge left, reverse back."""
    changed = False
    for r in range(4):
        row = (board[index_rc(r,3)], board[index_rc(r,2)],
               board[index_rc(r,1)], board[index_rc(r,0)])
        new_row, ch = ROW_MAP[row]
        if ch:
            changed = True
        board[index_rc(r,3)] = new_row[0]
        board[index_rc(r,2)] = new_row[1]
        board[index_rc(r,1)] = new_row[2]
        board[index_rc(r,0)] = new_row[3]
    return changed

def move_up_1d(board):
    """In-place 'move up' = transpose columns into rows, merge left, transpose back."""
    changed = False
    for c in range(4):
        col = (board[index_rc(0,c)], board[index_rc(1,c)],
               board[index_rc(2,c)], board[index_rc(3,c)])
        new_col, ch = ROW_MAP[col]
        if ch:
            changed = True
        board[index_rc(0,c)] = new_col[0]
        board[index_rc(1,c)] = new_col[1]
        board[index_rc(2,c)] = new_col[2]
        board[index_rc(3,c)] = new_col[3]
    return changed

def move_down_1d(board):
    """In-place 'move down' = transpose columns, merge right, transpose back."""
    changed = False
    for c in range(4):
        col = (board[index_rc(3,c)], board[index_rc(2,c)],
               board[index_rc(1,c)], board[index_rc(0,c)])
        new_col, ch = ROW_MAP[col]
        if ch:
            changed = True
        board[index_rc(3,c)] = new_col[0]
        board[index_rc(2,c)] = new_col[1]
        board[index_rc(1,c)] = new_col[2]
        board[index_rc(0,c)] = new_col[3]
    return changed

def apply_move_1d(board, move_id):
    if move_id == 0:
        return move_up_1d(board)
    elif move_id == 1:
        return move_right_1d(board)
    elif move_id == 2:
        return move_down_1d(board)
    return move_left_1d(board)

def get_legal_moves_1d(board):
    moves = []
    for m in range(4):
        temp = board[:]
        changed = apply_move_1d(temp, m)
        if changed:
            moves.append(m)
    return moves


def random_survival_rollout(board):
    sim = board[:]
    move_count = 0
    while True:
        if is_game_over_1d(sim):
            break
        moves = get_legal_moves_1d(sim)
        if not moves:
            break
        choice = random.choice(moves)
        changed = apply_move_1d(sim, choice)
        if changed:
            spawn_tile_1d(sim)
            move_count += 1
    return move_count


class MCTSNode:
    def __init__(self, board, parent=None, move_from_parent=None):
        self.board = board[:]
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = []
        self.untried_moves = get_legal_moves_1d(self.board)
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
                avg = child.total_value / child.visits
                explore = c * math.sqrt(2.0 * math.log(self.visits) / child.visits)
                ucb = avg + explore

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child
        return best_node

    def add_child(self, move_id, board_after_move):
        child = MCTSNode(board_after_move, parent=self, move_from_parent=move_id)
        self.children.append(child)
        self.untried_moves.remove(move_id)
        return child

    def update(self, value):
        self.visits += 1
        self.total_value += value

# -----------------------------------------------------------
#  MCTS Methods
# -----------------------------------------------------------

def tree_policy(node, c=1.41):
    while True:
        if is_game_over_1d(node.board):
            return node
        if not node.is_fully_expanded():
            return expand(node)
        node = node.best_child(c)

def expand(node):
    m = random.choice(node.untried_moves)
    temp = node.board[:]
    changed = apply_move_1d(temp, m)
    if changed:
        spawn_tile_1d(temp)
        return node.add_child(m, temp)
    node.untried_moves.remove(m)
    return node

def default_policy(board):
    return random_survival_rollout(board)

def backup(node, value):
    while node is not None:
        node.update(value)
        node = node.parent

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


def mcts(board, iterations=1000, c=1.41):
    """
    1) Create root node
    2) For i in [0..iterations):
        a) node = tree_policy(root)
        b) value = default_policy(node.board)
        c) backup(node, value)
    3) Return best move from root
    """
    root = MCTSNode(board)
    for _ in range(iterations):
        leaf = tree_policy(root, c)
        value = default_policy(leaf.board)
        backup(leaf, value)

    # pick best child of root by average survival
    best_child_node = None
    best_avg = float('-inf')
    for child in root.children:
        if child.visits > 0:
            avg_survival = child.total_value / child.visits
            if avg_survival > best_avg:
                best_avg = avg_survival
                best_child_node = child
    tree_visualization = visualize_tree(root, max_nodes=200)
    tree_visualization.render("decision_tree/mcts_cuda/mcts_tree.png", format="png", cleanup=True)  # Saves as 'mcts_tree.png'
    tree_visualization                
    return best_child_node.move_from_parent if best_child_node else None


if __name__ == "__main__":
    random.seed(42)
    # Create an empty 16-cell board
    board_1d = [0]*16
    spawn_tile_1d(board_1d)
    spawn_tile_1d(board_1d)

    print("Initial Board (exponents):", board_1d)
    best_move_id = mcts(board_1d, iterations=2000, c=1.41)
    moves = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT"}
    print("Recommended Move:", moves.get(best_move_id, "NONE"))
    

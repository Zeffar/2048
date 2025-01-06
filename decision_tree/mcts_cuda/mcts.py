import math
import random

# ------------------ Precomputed Scores ------------------
TABLE_SCORE = {
    1: 0,       # 2
    2: 4,       # 4
    3: 16,      # 8
    4: 48,      # 16
    5: 128,     # 32
    6: 320,     # 64
    7: 768,     # 128
    8: 1792,    # 256
    9: 4096,    # 512
    10: 9216,   # 1024
    11: 20480,  # 2048
    12: 45056,  # 4096
    13: 98304   # 8192
}
MAX_POSSIBLE_SCORE = max(TABLE_SCORE.values())  # 98304

# ------------------ 1D Board Tools ------------------
def index_rc(r, c):
    """Return the 1D index for row=r, col=c in a 4x4 board."""
    return (r << 2) + c  #  (r*4 + c), using bit-shift is slightly faster

def get_val(board, r, c):
    return board[index_rc(r, c)]

def set_val(board, r, c, value):
    board[index_rc(r, c)] = value

def compute_score_1d(board):
    """Sum up TABLE_SCORE[tile_exponent]."""
    s = 0
    for val in board:
        s += TABLE_SCORE.get(val, 0)
    return s

# --- Move-Lookup Precomputation (Row Merge) ---
# We generate a dictionary so that "move_left" for a 4-tuple is O(1).
# For every 4-tuple (r0,r1,r2,r3), we compute the new row + whether it changed.
row_merge_map = {}

def merge_left_4(row):
    """
    Return (new_row, changed).
    row is a tuple (4 elements).
    new_row is a tuple after 2048-like merge to the left.
    """
    r = list(row)
    changed = False
    # Merge
    for i in range(3):
        if r[i] != 0 and r[i] == r[i+1]:
            r[i] += 1
            r[i+1] = 0
            changed = True
    # Compact
    merged = [x for x in r if x != 0]
    merged += [0]*(4 - len(merged))
    if tuple(merged) != row:
        changed = True
    return tuple(merged), changed

# Fill row_merge_map
for r0 in range(15):   # 0..14 or maybe bigger if you want tiles beyond exponent=13
    for r1 in range(15):
        for r2 in range(15):
            for r3 in range(15):
                original = (r0, r1, r2, r3)
                row_merge_map[original] = merge_left_4(original)

def move_left_1d(board):
    """Perform 'move left' in-place for a 1D board of length 16. Return bool 'changed'."""
    changed = False
    # For each row r
    for r in range(4):
        # Extract row
        row_tuple = ( board[index_rc(r,0)], board[index_rc(r,1)],
                      board[index_rc(r,2)], board[index_rc(r,3)] )
        new_row, row_changed = row_merge_map[row_tuple]
        if row_changed:
            changed = True
        # Write back
        board[index_rc(r,0)] = new_row[0]
        board[index_rc(r,1)] = new_row[1]
        board[index_rc(r,2)] = new_row[2]
        board[index_rc(r,3)] = new_row[3]
    return changed

def move_right_1d(board):
    """Move right = reverse row, merge left, reverse row."""
    changed = False
    for r in range(4):
        # Extract row
        row_tuple = ( board[index_rc(r,3)], board[index_rc(r,2)],
                      board[index_rc(r,1)], board[index_rc(r,0)] )
        new_row, row_changed = row_merge_map[row_tuple]
        if row_changed:
            changed = True
        # Write back reversed
        board[index_rc(r,3)] = new_row[0]
        board[index_rc(r,2)] = new_row[1]
        board[index_rc(r,1)] = new_row[2]
        board[index_rc(r,0)] = new_row[3]
    return changed

def move_up_1d(board):
    """Move up = transpose, move_left, transpose."""
    changed = False
    # We'll handle columns as rows: c in [0..3]
    for c in range(4):
        col_tuple = ( board[index_rc(0,c)], board[index_rc(1,c)],
                      board[index_rc(2,c)], board[index_rc(3,c)] )
        new_col, row_changed = row_merge_map[col_tuple]
        if row_changed:
            changed = True
        board[index_rc(0,c)] = new_col[0]
        board[index_rc(1,c)] = new_col[1]
        board[index_rc(2,c)] = new_col[2]
        board[index_rc(3,c)] = new_col[3]
    return changed

def move_down_1d(board):
    """Move down = transpose, move_right, transpose."""
    changed = False
    for c in range(4):
        col_tuple = ( board[index_rc(3,c)], board[index_rc(2,c)],
                      board[index_rc(1,c)], board[index_rc(0,c)] )
        new_col, row_changed = row_merge_map[col_tuple]
        if row_changed:
            changed = True
        board[index_rc(3,c)] = new_col[0]
        board[index_rc(2,c)] = new_col[1]
        board[index_rc(1,c)] = new_col[2]
        board[index_rc(0,c)] = new_col[3]
    return changed

def get_legal_moves_1d(board):
    """Return list of moves [0..3] that change the board."""
    moves = []
    # We'll clone the board once and do each move
    for move_id in range(4):
        clone = board[:]
        changed = False
        if move_id == 0:
            changed = move_up_1d(clone)
        elif move_id == 1:
            changed = move_right_1d(clone)
        elif move_id == 2:
            changed = move_down_1d(clone)
        else:
            changed = move_left_1d(clone)
        if changed:
            moves.append(move_id)
    return moves

def apply_move_1d(board, move_id):
    """In-place move (0=UP,1=RIGHT,2=DOWN,3=LEFT). Return bool changed."""
    if move_id == 0:
        return move_up_1d(board)
    elif move_id == 1:
        return move_right_1d(board)
    elif move_id == 2:
        return move_down_1d(board)
    else:
        return move_left_1d(board)

def spawn_tile_1d(board):
    """Spawn a '2' (exponent=1) in a random empty cell."""
    empty = []
    for i in range(16):
        if board[i] == 0:
            empty.append(i)
    if not empty:
        return
    idx = random.choice(empty)
    board[idx] = 1

def is_game_over_1d(board):
    """Check if there are no legal moves."""
    # Quick check: if any empty cell, not over
    for i in range(16):
        if board[i] == 0:
            return False
    # If no empty, see if merging is possible in any direction
    # Instead of calling get_legal_moves_1d, we can do minimal checks:
    # Check horizontally
    for r in range(4):
        for c in range(3):
            i1 = index_rc(r, c)
            i2 = index_rc(r, c+1)
            if board[i1] == board[i2]:
                return False
    # Check vertically
    for c in range(4):
        for r in range(3):
            i1 = index_rc(r, c)
            i2 = index_rc(r+1, c)
            if board[i1] == board[i2]:
                return False
    return True


# ------------------ MCTS Node ------------------
class MCTSNode:
    def __init__(self, board_1d, parent=None, move_from_parent=None):
        # We'll store a copy of the board so the node is stable.
        self.board = board_1d[:]  # Make a shallow copy
        self.parent = parent
        self.move_from_parent = move_from_parent

        self.children = []
        self.untried_moves = get_legal_moves_1d(self.board)

        self.visits = 0
        self.total_value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.41):
        """
        UCB1 with normalized average:
          normalized_avg = (child.total_value / child.visits) / MAX_POSSIBLE_SCORE
        """
        best_ucb = float('-inf')
        best_node = None

        for child in self.children:
            if child.visits == 0:
                return child
            avg_score = child.total_value / child.visits
            normalized_avg = avg_score / MAX_POSSIBLE_SCORE

            # UCB
            exploit = normalized_avg
            explore = c * math.sqrt(2.0 * math.log(self.visits) / child.visits)
            ucb = exploit + explore

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child
        return best_node

    def add_child(self, move_id, new_board):
        child = MCTSNode(new_board, parent=self, move_from_parent=move_id)
        self.children.append(child)
        self.untried_moves.remove(move_id)
        return child

    def update(self, value):
        self.visits += 1
        self.total_value += value

# ------------------ MCTS Core Functions ------------------
def tree_policy(node):
    """Selection + expansion until terminal."""
    while not is_game_over_1d(node.board):
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = node.best_child()
    return node

def expand(node):
    """Pick an untried move at random, apply, spawn tile, create child."""
    m = random.choice(node.untried_moves)
    new_board = node.board[:]
    changed = apply_move_1d(new_board, m)
    if not changed:
        # theoretically shouldn't happen if untried_moves is correct
        node.untried_moves.remove(m)
        return node
    spawn_tile_1d(new_board)
    return node.add_child(m, new_board)

def default_policy(board_1d):
    """Rollout: random moves until game over. Return final score."""
    # We'll do an in-place simulation with a copy of the board
    sim_board = board_1d[:]
    while not is_game_over_1d(sim_board):
        moves = get_legal_moves_1d(sim_board)
        if not moves:
            break
        move_id = random.choice(moves)
        apply_move_1d(sim_board, move_id)
        spawn_tile_1d(sim_board)
    return compute_score_1d(sim_board)

def backup(node, value):
    """Propagate result."""
    cur = node
    while cur is not None:
        cur.update(value)
        cur = cur.parent

from graphviz import Digraph

def visualize_tree(root, max_nodes=4000):
    """
    Visualize the MCTS tree structure in a layer-by-layer fashion using graphviz.
    Each layer is limited to at most 16 nodes (based on the best metric).
    
    The 'best' 16 children are chosen by descending (total_value / visits).
    If visits=0, we treat the metric as 0.0.
    
    Args:
        root: The root MCTSNode of the tree.
        max_nodes: Maximum number of total nodes to include (prevents overload).
    """
    dot = Digraph(comment="MCTS Tree")

    # For easy node labeling in the graph
    def make_label(node):
        label = f"Visits: {node.visits}\nValue: {node.total_value / node.visits:.2f}"
        if node.move_from_parent is not None:
            label += f"\nMove: {node.move_from_parent}"
        return label

    # Assign the root an ID and add it to the graph
    node_id_map = {}
    node_id_map[root] = 0
    dot.node("0", label=make_label(root))

    # 'current_layer' will be a list of nodes for the current depth
    current_layer = [root]
    layer = 0

    # We'll continue until we run out of layers or exceed max_nodes
    while current_layer and len(node_id_map) < max_nodes:
        next_layer_candidates = []
        
        # Gather all children for the entire layer
        for node in current_layer:
            # Collect the children of this node
            for child in node.children:
                # If we've reached max capacity, break
                if len(node_id_map) >= max_nodes:
                    break

                # If the child isn't already in the map (not yet visualized),
                # prepare it as a candidate for the next layer.
                if child not in node_id_map:
                    # Compute the metric (total_value / visits)
                    if child.visits > 0:
                        metric = child.total_value / child.visits
                    else:
                        metric = 0.0  # or skip it entirely

                    parent_id = node_id_map[node]
                    next_layer_candidates.append((child, parent_id, metric))
        
        # We have now gathered all candidates for the next layer
        if not next_layer_candidates:
            break

        # Sort descending by metric and take the top 16
        next_layer_candidates.sort(key=lambda x: x[2], reverse=True)
        next_layer_candidates = next_layer_candidates[:16]

        # Now we create the new layer from the chosen candidates
        next_layer = []
        for child, parent_id, metric in next_layer_candidates:
            if child not in node_id_map:
                new_id = len(node_id_map)
                node_id_map[child] = new_id
                dot.node(str(new_id), label=make_label(child))
                dot.edge(str(parent_id), str(new_id))
                next_layer.append(child)
            else:
                # The child was already in the map (rare in a pure tree),
                # but if it happens, just ensure we connect the edge
                # if it's not yet connected.
                child_id = node_id_map[child]
                dot.edge(str(parent_id), str(child_id))
        
        # Move to the next layer
        current_layer = next_layer
        layer += 1

    return dot


def mcts(board_1d, iterations=1000, c=1.41):
    root = MCTSNode(board_1d)
    for _ in range(iterations):
        leaf = tree_policy(root)
        reward = default_policy(leaf.board)
        backup(leaf, reward)

    best_child_node = root.best_child(c=c)
    tree_visualization = visualize_tree(root, max_nodes=2000)
    tree_visualization.render("mcts_tree", format="png", cleanup=True)  # Saves as 'mcts_tree.png'
    tree_visualization
    return best_child_node.move_from_parent

# ------------------ Example Usage ------------------
if __name__ == "__main__":
    # Initialize a 1D board, place 2 tiles
    board_1d = [0]*16
    spawn_tile_1d(board_1d)
    spawn_tile_1d(board_1d)

    print("Initial board (1D):", board_1d)

    # MCTS
    best_move = mcts(board_1d, iterations=10000)
    move_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    print("MCTS recommends:", move_map.get(best_move, "?"))

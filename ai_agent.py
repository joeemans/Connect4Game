# import math
# import random
# import time
# from typing import List, Tuple
# import json 
# import os # <--- NEW IMPORT

# from game import Connect4Game

# # Define the filename for the search tree JSON
# TREE_FILE_NAME = "search_tree.json"

# class Node:
#     # Node class - data structure
#     def __init__(self, state, parent=None, move=None, node_type="decision", probability=1.0):
#         self.state = state
#         self.parent = parent
#         self.children = []
#        # self.alpha = float('-inf')
#        # self.beta = float('inf')
#         self.move = move # The move that led to this state
#         self.heuristic_value = None
#         self.node_type = node_type # decision or chance
#         self.probability = probability

#     def add_child(self, child_node):
#         self.children.append(child_node)
    
#     def get_depth(self):
#         depth = 0
#         current_node = self
#         while current_node.parent is not None:
#             depth += 1
#             current_node = current_node.parent
#         return depth
    

# class AIAgent:  
#     def __init__(self):
#         # initialize AI's models or parameters here
#         #self.game = game
#         self.heuristic_enabled = False
#         self.last_root = None
#         self.reset_statistics() # Initialize statistics

#     def set_heuristic_enabled(self, enabled: bool):
#         # Toggle heuristic evaluation on or off
#         self.heuristic_enabled = bool(enabled)

#     def get_last_tree(self):
#         # Expose the most recently generated search tree root
#         return self.last_root

#     def reset_statistics(self):
#         self.nodes_expanded = 0
#         self.max_depth_reached = 0
#         self.pruned_nodes = 0
#         self.working_time = 0

#     def _copy_game_state(self, game):
#         new_game = Connect4Game(game.rows, game.cols)
#         new_game.board = game.board.copy()
#         new_game.humanScore = game.humanScore
#         new_game.aiScore = game.aiScore
#         new_game.winner = game.winner
#         new_game.turn = game.turn
#         return new_game

#     def minimax(self, node, depth, maximizing_player = True):
#         # Minimax algorithm without pruning
#         self.max_depth_reached = max(self.max_depth_reached, depth)
#         game = node.state # Current game state

#         if game.is_terminal_node() or depth == 0: 
#             node.heuristic_value = self._evaluate_state(game)
#             return node.heuristic_value

#         self.nodes_expanded += 1 # Count this node as expanded -> we will look for its children
#         valid_moves = game.get_valid_moves()

#         if maximizing_player: # AI's turn
#             value = -math.inf
#             for col in valid_moves: # DLS through all valid moves
#                 new_game = self._copy_game_state(game) # Create copy of the game state so we can simulate moves
#                 new_game.drop_piece(col, new_game.AI)
#                 child_node = Node(new_game, parent=node, move=col) # Create child node
#                 node.add_child(child_node)
#                 score = self.minimax(child_node, depth - 1, False)
#                 value = max(value, score) # Get max value from successors
            
#             node.heuristic_value = value # Update node's heuristic value
#             return value

#         else: # Minimizing player's turn (Simulating human moves)
#             value = math.inf
#             for col in valid_moves:
#                 new_game = self._copy_game_state(game)
#                 new_game.drop_piece(col, new_game.HUMAN)
#                 child_node = Node(new_game, parent=node, move=col)
#                 node.add_child(child_node)
#                 score = self.minimax(child_node, depth - 1, True)
#                 value = min(value, score)
                
#             node.heuristic_value = value
#             return value

#     def minimax_alpha_beta(self, node, depth, alpha, beta, maximizing_player = True):
#         # Minimax algorithm with alpha-beta pruning
#         self.max_depth_reached = max(self.max_depth_reached, depth)
#         game = node.state

#         if game.is_terminal_node() or depth == 0:
#             node.heuristic_value = self._evaluate_state(game)
#             return node.heuristic_value

#         self.nodes_expanded += 1 # Count this node as expanded
#         valid_moves = game.get_valid_moves()

#         if maximizing_player:
#             value = -math.inf
#             for move_index, col in enumerate(valid_moves):
#                 new_game = self._copy_game_state(game)
#                 new_game.drop_piece(col, new_game.AI)
#                 child_node = Node(new_game, parent=node, move=col)
#                 node.add_child(child_node)
#                 score = self.minimax_alpha_beta(child_node, depth - 1, alpha, beta, False)
#                 value = max(value, score)
#                 alpha = max(alpha, value)
#                 if alpha >= beta:
#                     pruned_here = len(valid_moves) - (move_index + 1) # Remaining moves pruned. Move index added specifically for that
#                     self.pruned_nodes += pruned_here
#                     break
#             node.heuristic_value = value
#             return value
#         else:
#             value = math.inf
#             for move_index, col in enumerate(valid_moves):
#                 new_game = self._copy_game_state(game)
#                 new_game.drop_piece(col, new_game.HUMAN)
#                 child_node = Node(new_game, parent=node, move=col)
#                 node.add_child(child_node)
#                 score = self.minimax_alpha_beta(child_node, depth - 1, alpha, beta, True)
#                 value = min(value, score)
#                 beta = min(beta, value)
#                 if alpha >= beta:
#                     pruned_here = len(valid_moves) - (move_index + 1)
#                     self.pruned_nodes += pruned_here
#                     break
#             node.heuristic_value = value
#             return value

#     def expectiminimax(self, node, depth, maximizing_player = True):
#         # Expectiminimax algorithm to account for potential slips
#         self.max_depth_reached = max(self.max_depth_reached, depth)
#         game = node.state

#         if game.is_terminal_node() or depth == 0:
#             node.heuristic_value = self._evaluate_state(game)
#             return node.heuristic_value

#         self.nodes_expanded += 1 # Count this node as expanded
#         valid_moves = game.get_valid_moves()
#         if not valid_moves:
#             node.heuristic_value = self._evaluate_state(game)
#             return node.heuristic_value

#         if maximizing_player:
#             best_value = -math.inf
#             for col in valid_moves:
#                 # For each MAX decision, go through a CHANCE layer before the MIN decision
#                 expected_value = self._evaluate_expected_outcome(node, game, col, depth, True)
#                 best_value = max(best_value, expected_value)
#             node.heuristic_value = best_value
#             return best_value
#         else:
#             best_value = math.inf
#             for col in valid_moves:
#                 # For each MIN decision, we add a chance layer
#                 expected_value = self._evaluate_expected_outcome(node, game, col, depth, False)
#                 best_value = min(best_value, expected_value)
#             node.heuristic_value = best_value
#             return best_value
 
#     def get_move(self, game, depth_k, algorithm_type):
#         # Entry point for AI to get its move
#         print(f"\nAI is thinking (Algorithm: {algorithm_type}, Depth: {depth_k})...")
        
#         # Reset stats and start timer
#         self.reset_statistics()
#         start_time = time.time()
        
#         # Create the root node
#         root_game_state = self._copy_game_state(game) # IMPORTANT: We are using a deep copy so the original game object is not changed
#         root_node = Node(root_game_state, move=-1) # Root has no move leading to it
#         self.last_root = root_node

#         if algorithm_type == "minimax":
#             best_score = self.minimax(root_node, depth_k, True)
#         elif algorithm_type == "alpha_beta":
#             best_score = self.minimax_alpha_beta(root_node, depth_k, -math.inf, math.inf, True)
#         elif algorithm_type == "expectiminimax":
#             best_score = self.expectiminimax(root_node, depth_k, True)
#         elif algorithm_type == "random":
#             return self.get_random_move(root_game_state)
#         else:
#             raise ValueError("Unknown algorithm type")

#         # Stop timer and calculate duration
#         end_time = time.time()
#         self.working_time = end_time - start_time
        
#         best_move_col = None
#         for child in root_node.children:
#             if child.heuristic_value == best_score:
#                 best_move_col = child.move
#                 break
        
#         # Failsafe: if no move matches, pick a random one
#         if best_move_col is None:
#             raise Exception("No valid move found by AI. This should not happen.")

#         # Export the Tree's JSON to a file
#         tree_json_data = self.to_json_structure(root_node)
#         try:
#             with open(TREE_FILE_NAME, 'w') as f:
#                 json.dump(tree_json_data, f, indent=2)
#             print(f"\n--- Game Tree Exported ---")
#             print(f"Tree structure saved to {TREE_FILE_NAME}")
#             print(f"Open 'tree_visualizer.html' and click 'Load Tree' to view the visualization.")
#             print("--------------------------")
#         except IOError as e:
#             print(f"ERROR: Could not write tree JSON to file {TREE_FILE_NAME}. {e}")
#             print("Printing to console as fallback:")
#             print(json.dumps(tree_json_data, indent=2))
        
#         # # Print the Tree
#         # print("\n--- Game Tree ---")
#         # self.print_tree(root_node)
               
#         # Print Stats
#         print("--- AI Move Statistics ---")
#         print(f"Time Taken: {self.working_time:.4f} seconds")
#         print(f"Nodes Expanded: {self.nodes_expanded}")
#         print(f"Max Depth Reached: {self.max_depth_reached} from start node")
#         print(f"Nodes Pruned (Alpha-Beta): {self.pruned_nodes}")
#         print(f"Best Score Found: {best_score}")
#         print(f"Chosen Move (Column): {best_move_col}")

#         return best_move_col

#     def print_tree(self, node, depth=0):
#         prefix = " " * depth * 2
#         segments = [f"Move: {node.move}", f"Type: {node.node_type}", f"Value: {node.heuristic_value}"]
#         if node.probability is not None and (node.node_type == "chance" or abs(node.probability - 1.0) > 1e-9):
#             segments.append(f"Prob: {round(node.probability, 2)}")
#         print(prefix + ", ".join(segments))
#         for child in node.children:
#             self.print_tree(child, depth + 1)

#     def to_json_structure(self, node, is_maximizing=None):
#         """Recursively converts a Node structure to a serializable dictionary (JSON structure)."""
#         if node is None:
#             return {}

#         node_type = getattr(node, "node_type", "decision")
#         move = getattr(node, "move", -1)
#         value = getattr(node, "heuristic_value", None)
#         probability = getattr(node, "probability", 1.0)
        
#         # Determine the player for Decision nodes
#         player = None
#         if node_type == "decision":
#             # Using the absolute depth is simpler for labeling in the JSON:
#             is_maximizing = (node.get_depth() % 2 == 0) # Assumes alternating MAX/MIN turns starting with MAX at depth 0
            
#             player = "MAX (AI)" if is_maximizing else "MIN (Human)"
        
#         # For chance nodes, the player is not a factor.
#         elif node_type == "chance":
#             player = "Chance"

#         data = {
#             "move": move,
#             "type": node_type,
#             "player": player,
#             "value": round(value, 4) if value is not None else "N/A",
#             "probability": round(probability, 4) if probability != 1.0 else 1.0,
#         }
        
#         children_list = []
#         # Optimization: Limit recursion for JSON structure to prevent massive output file sizes
#         # The limit is set to Depth 6.
#         if node.get_depth() <= 6:
#             for child in node.children:
#                 children_list.append(self.to_json_structure(child))
        
#         data["children"] = children_list

#         # Note if the branch was stopped for size reasons
#         if node.get_depth() == 6 and len(node.children) > 0: 
#             data["truncated"] = True
            
#         return data

#     def print_statistics(self):
#         print(f"Nodes Expanded: {self.nodes_expanded}")
#         print(f"Max Depth Reached: {self.max_depth_reached}")
#         print(f"Pruned Nodes: {self.pruned_nodes}")
#         print(f"Working Time: {self.working_time} seconds")


#     def get_random_move(self, game):  
#         # Get all possible moves
#         valid_moves = game.get_valid_moves()
        
#         # Return a random choice from the list
#         if valid_moves:
#             return random.choice(valid_moves)
        
#         return None # Should not happen if game isn't over

#     def _evaluate_state(self, game):
#         # Choose between pure score evaluation and heuristic-based evaluation
#         if self.heuristic_enabled:
#             return self._evaluate_with_heuristic(game)
#         return self._evaluate_score_only(game)

#     def _evaluate_score_only(self, game):
#         # Use the actual score difference (AI - Human)
#         human_score, ai_score = game.count_connected_fours()
#         return ai_score - human_score

#     def _evaluate_with_heuristic(self, game):
#         # Heuristic evaluation. Our heuristic is divided to:
#         # 1. Completed connected-fours (weighted heavily)
#         # 2. Potential threats (3-in-a-rows, 2-in-a-rows, 1-in-a-rows)
#         # 3. Center / central columns occupation (slight preference for central pieces)
#         # 4. Blocking opponent's threats (negative weight, symmetric to own threats)

#         board = game.board
#         ROWS, COLS = board.shape

#         # Completed connected-fours (scaled by 1000 as requested)
#         human_score, ai_score = game.count_connected_fours()
#         score = (ai_score - human_score) * 1000

#         # Build a column weight map: center columns get highest weight, borders get 1.
#         col_weights = self._build_column_weights(COLS)

#         # Build a rows weight map: similar to columns
#         row_weights = self._build_column_weights(ROWS)

#         # Add weight maps
#         score += self._score_positional_map(board, game.AI, col_weights, row_weights)
#         score -= self._score_positional_map(board, game.HUMAN, col_weights, row_weights)

#         # We evaluate windows from AI perspective and subtract symmetric values for HUMAN.
#         score += self._score_threats(board, game.AI)
#         score -= self._score_threats(board, game.HUMAN)

#         return score

    
#     def _build_column_weights(self, cols: int):
#         # Assign higher weights to central columns and weight 1 to border columns.
#         # Central column(s) get weight = cols/2, weights decrease stepwise toward borders.

        
#         if cols <= 0:
#             return []
#         max_weight = cols / 2.0  # Weight of the most central column(s)
#         center = (cols - 1) / 2.0 # Center index (can be non-integer for even cols)
#         weights = []
#         for c in range(cols):
#             distance = abs(c - center) # distance from center
#             if max_weight <= 1:
#                 weights.append(1.0) # should not happen ever anyways
#             else:
#                 # Linear drop: weight = max_weight - k * distance, clipped to [1, max_weight]
#                 # Choose k so that farthest column has weight ~= 1.
#                 max_distance = center # max distance from center to border
#                 if (max_distance == 0): # safety check just in case - should never happen
#                     weights.append(1.0)
#                     print("Warning: max_distance is zero in _build_column_weights.")
#                     continue
#                 k = (max_weight - 1.0) / max_distance # slope
#                 w = max_weight - k * distance # this makes most central column weight = max_weight, borders = 1
#                 if w < 1.0:
#                     w = 1.0
#                     print("Warning: computed weight less than 1.0 in _build_column_weights.")
#                 weights.append(w)
#         return weights

#     def _build_row_weights(self, rows: int):
#         # Assign higher weights to central rows and weight 1 to border rows.
#         # Central row(s) get weight = rows/2, weights decrease stepwise toward borders.

        
#         if rows <= 0:
#             return []
#         max_weight = rows / 2.0  # Weight of the most central row(s)
#         center = (rows - 1) / 2.0 # Center index (can be non-integer for even rows)
#         weights = []
#         for r in range(rows):
#             distance = abs(r - center) # distance from center
#             if max_weight <= 1:
#                 weights.append(1.0) # should not happen ever anyways
#             else:
#                 # Linear drop: weight = max_weight - k * distance, clipped to [1, max_weight]
#                 # Choose k so that farthest row has weight ~= 1.
#                 max_distance = center # max distance from center to border
#                 if (max_distance == 0): # safety check just in case - should never happen
#                     weights.append(1.0)
#                     print("Warning: max_distance is zero in _build_row_weights.")
#                     continue
#                 k = (max_weight - 1.0) / max_distance # slope
#                 w = max_weight - k * distance # this makes most central row weight = max_weight, borders = 1
#                 if w < 1.0:
#                     w = 1.0
#                     print("Warning: computed weight less than 1.0 in _build_row_weights.")
#                 weights.append(w)
#         return weights
    
#     def _score_positional_map(self, board, piece, col_weights, row_weights):
#         score = 0.0
#         rows, cols = board.shape
#         for r in range(rows):
#             for c in range(cols):
#                 if board[r][c] == piece:
#                     score += col_weights[c] * 0.5
#                     score += row_weights[r] * 0.25 # rows are less important than columns
#         return score
    
#     def _score_threats(self, board, piece):
#         # Scores all 4-cell windows based purely on piece count.

#         score = 0
#         ROWS, COLS = board.shape
#         opponent = 1 if piece == 2 else 2 # opposite of player - aka piece

#         # helper to score a single window
#         def evaluate_window(window):
#             count_piece = window.count(piece)
#             count_empty = window.count(0) # empty tiles
#             count_opp = window.count(opponent)

#             # If window is blocked by opponent, it has 0 potential -> 0
#             if count_opp > 0:
#                 return 0
            
#             # Scoring weights
#             if count_piece == 3 and count_empty == 1:
#                 return 100  # Strong Threat
#             elif count_piece == 2 and count_empty == 2:
#                 return 5    # Minor Threat
            
#             return 0

#         # Horizontal
#         for r in range(ROWS):
#             row_array = list(board[r, :])
#             for c in range(COLS - 3):
#                 score += evaluate_window(row_array[c:c+4])

#         # Vertical
#         for c in range(COLS):
#             col_array = list(board[:, c])
#             for r in range(ROWS - 3):
#                 score += evaluate_window(col_array[r:r+4])

#         # Positive Slope Diagonal
#         for r in range(ROWS - 3):
#             for c in range(COLS - 3):
#                 window = [board[r+i][c+i] for i in range(4)]
#                 score += evaluate_window(window)

#         # Negative Slope Diagonal
#         for r in range(3, ROWS):
#             for c in range(COLS - 3):
#                 window = [board[r-i][c+i] for i in range(4)]
#                 score += evaluate_window(window)

#         return score

#     def _evaluate_expected_outcome(self, parent_node, game, selected_col, depth, is_ai_turn):
#         # Build expectation node for the selected column and compute expected heuristic value
#         selection_node = Node(self._copy_game_state(game), parent=parent_node, move=selected_col, node_type="chance")
#         parent_node.add_child(selection_node)

#         probabilities = self._get_slip_probabilities(game, selected_col)
#         piece = game.AI if is_ai_turn else game.HUMAN

#         expected_value = 0.0
#         total_weight = 0.0

#         # Note: The is_maximizing flag logic needs to be correctly managed here for the recursive call.
#         # Since this is a chance layer, the next decision is the opposite of the player who just moved.
#         # next_is_maximizing = not is_ai_turn

#         for actual_col, probability in probabilities:
#             if not game.is_valid_move(actual_col):
#                 continue
#             next_state = self._copy_game_state(game)
#             next_state.drop_piece(actual_col, piece)
#             outcome_node = Node(next_state, parent=selection_node, move=actual_col, probability=probability)
#             selection_node.add_child(outcome_node)
            
#             # Call expectiminimax which correctly handles MAX/MIN based on the flag
#             child_value = self.expectiminimax(outcome_node, depth - 1, not is_ai_turn)
#             expected_value += probability * child_value
#             total_weight += probability

#         if total_weight == 0:
#             selection_node.heuristic_value = self._evaluate_state(game)
#             return selection_node.heuristic_value

#         selection_node.heuristic_value = expected_value / total_weight
#         return selection_node.heuristic_value

#     def _get_slip_probabilities(self, game, selected_col):
#         # Determine the probability distribution for where a piece actually lands
#         probabilities: List[Tuple[int, float]] = []
#         base_prob = 0.6 if game.is_valid_move(selected_col) else 0.0
#         if base_prob > 0:
#             probabilities.append((selected_col, base_prob))

#         left_col = selected_col - 1
#         right_col = selected_col + 1
#         left_valid = left_col >= 0 and game.is_valid_move(left_col)
#         right_valid = right_col < game.cols and game.is_valid_move(right_col)

#         if left_valid and right_valid:
#             probabilities.append((left_col, 0.2))
#             probabilities.append((right_col, 0.2))
#         elif left_valid:
#             probabilities.append((left_col, 0.4))
#         elif right_valid:
#             probabilities.append((right_col, 0.4))

#         total = sum(weight for _, weight in probabilities)
#         if total == 0:
#             return []

#         return [(col, weight / total) for col, weight in probabilities]

import math
import random
import time
from typing import List, Tuple
import json 
import os # <--- NEW IMPORT

from game import Connect4Game

# Define the filename for the search tree JSON
TREE_FILE_NAME = "search_tree.json"

class Node:
    # Node class - data structure
    def __init__(self, state, parent=None, move=None, node_type="decision", probability=1.0):
        self.state = state
        self.parent = parent
        self.children = []
       # self.alpha = float('-inf')
       # self.beta = float('inf')
        self.move = move # The move that led to this state
        self.heuristic_value = None
        self.node_type = node_type # decision or chance
        self.probability = probability

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def get_depth(self):
        depth = 0
        current_node = self
        while current_node.parent is not None:
            depth += 1
            current_node = current_node.parent
        return depth
    

class AIAgent:  
    def __init__(self):
        # initialize AI's models or parameters here
        #self.game = game
        self.heuristic_enabled = False
        self.last_root = None
        self.reset_statistics() # Initialize statistics

    def set_heuristic_enabled(self, enabled: bool):
        # Toggle heuristic evaluation on or off
        self.heuristic_enabled = bool(enabled)

    def get_last_tree(self):
        # Expose the most recently generated search tree root
        return self.last_root

    def reset_statistics(self):
        self.nodes_expanded = 0
        self.max_depth_reached = 0
        self.pruned_nodes = 0
        self.working_time = 0

    def _copy_game_state(self, game):
        new_game = Connect4Game(game.rows, game.cols)
        new_game.board = game.board.copy()
        new_game.humanScore = game.humanScore
        new_game.aiScore = game.aiScore
        new_game.winner = game.winner
        new_game.turn = game.turn
        return new_game

    def minimax(self, node, depth, maximizing_player = True):
        # Minimax algorithm without pruning
        self.max_depth_reached = max(self.max_depth_reached, depth)
        game = node.state # Current game state

        if game.is_terminal_node() or depth == 0: 
            node.heuristic_value = self._evaluate_state(game)
            return node.heuristic_value

        self.nodes_expanded += 1 # Count this node as expanded -> we will look for its children
        valid_moves = game.get_valid_moves()

        if maximizing_player: # AI's turn
            value = -math.inf
            for col in valid_moves: # DLS through all valid moves
                new_game = self._copy_game_state(game) # Create copy of the game state so we can simulate moves
                new_game.drop_piece(col, new_game.AI)
                child_node = Node(new_game, parent=node, move=col) # Create child node
                node.add_child(child_node)
                score = self.minimax(child_node, depth - 1, False)
                value = max(value, score) # Get max value from successors
            
            node.heuristic_value = value # Update node's heuristic value
            return value

        else: # Minimizing player's turn (Simulating human moves)
            value = math.inf
            for col in valid_moves:
                new_game = self._copy_game_state(game)
                new_game.drop_piece(col, new_game.HUMAN)
                child_node = Node(new_game, parent=node, move=col)
                node.add_child(child_node)
                score = self.minimax(child_node, depth - 1, True)
                value = min(value, score)
                
            node.heuristic_value = value
            return value

    def minimax_alpha_beta(self, node, depth, alpha, beta, maximizing_player = True):
        # Minimax algorithm with alpha-beta pruning
        self.max_depth_reached = max(self.max_depth_reached, depth)
        game = node.state

        if game.is_terminal_node() or depth == 0:
            node.heuristic_value = self._evaluate_state(game)
            return node.heuristic_value

        self.nodes_expanded += 1 # Count this node as expanded
        valid_moves = game.get_valid_moves()

        if maximizing_player:
            value = -math.inf
            for move_index, col in enumerate(valid_moves):
                new_game = self._copy_game_state(game)
                new_game.drop_piece(col, new_game.AI)
                child_node = Node(new_game, parent=node, move=col)
                node.add_child(child_node)
                score = self.minimax_alpha_beta(child_node, depth - 1, alpha, beta, False)
                value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    pruned_here = len(valid_moves) - (move_index + 1) # Remaining moves pruned. Move index added specifically for that
                    self.pruned_nodes += pruned_here
                    break
            node.heuristic_value = value
            return value
        else:
            value = math.inf
            for move_index, col in enumerate(valid_moves):
                new_game = self._copy_game_state(game)
                new_game.drop_piece(col, new_game.HUMAN)
                child_node = Node(new_game, parent=node, move=col)
                node.add_child(child_node)
                score = self.minimax_alpha_beta(child_node, depth - 1, alpha, beta, True)
                value = min(value, score)
                beta = min(beta, value)
                if alpha >= beta:
                    pruned_here = len(valid_moves) - (move_index + 1)
                    self.pruned_nodes += pruned_here
                    break
            node.heuristic_value = value
            return value

    def expectiminimax(self, node, depth, maximizing_player = True):
        # Expectiminimax algorithm to account for potential slips
        self.max_depth_reached = max(self.max_depth_reached, depth)
        game = node.state

        if game.is_terminal_node() or depth == 0:
            node.heuristic_value = self._evaluate_state(game)
            return node.heuristic_value

        self.nodes_expanded += 1 # Count this node as expanded
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            node.heuristic_value = self._evaluate_state(game)
            return node.heuristic_value

        if maximizing_player:
            best_value = -math.inf
            for col in valid_moves:
                # For each MAX decision, go through a CHANCE layer before the MIN decision
                expected_value = self._evaluate_expected_outcome(node, game, col, depth, True)
                best_value = max(best_value, expected_value)
            node.heuristic_value = best_value
            return best_value
        else:
            best_value = math.inf
            for col in valid_moves:
                # For each MIN decision, we add a chance layer
                expected_value = self._evaluate_expected_outcome(node, game, col, depth, False)
                best_value = min(best_value, expected_value)
            node.heuristic_value = best_value
            return best_value
 
    def get_move(self, game, depth_k, algorithm_type):
        # Entry point for AI to get its move
        print(f"\nAI is thinking (Algorithm: {algorithm_type}, Depth: {depth_k})...")
        
        # Reset stats and start timer
        self.reset_statistics()
        start_time = time.time()
        
        # Create the root node
        root_game_state = self._copy_game_state(game) # IMPORTANT: We are using a deep copy so the original game object is not changed
        root_node = Node(root_game_state, move=-1) # Root has no move leading to it
        self.last_root = root_node

        if algorithm_type == "minimax":
            best_score = self.minimax(root_node, depth_k, True)
        elif algorithm_type == "alpha_beta":
            best_score = self.minimax_alpha_beta(root_node, depth_k, -math.inf, math.inf, True)
        elif algorithm_type == "expectiminimax":
            best_score = self.expectiminimax(root_node, depth_k, True)
        elif algorithm_type == "random":
            return self.get_random_move(root_game_state)
        else:
            raise ValueError("Unknown algorithm type")

        # Stop timer and calculate duration
        end_time = time.time()
        self.working_time = end_time - start_time
        
        best_move_col = None
        for child in root_node.children:
            if child.heuristic_value == best_score:
                best_move_col = child.move
                break
        
        # Failsafe: if no move matches, pick a random one
        if best_move_col is None:
            raise Exception("No valid move found by AI. This should not happen.")

        # Export the Tree's JSON to a file
        tree_json_data = self.to_json_structure(root_node)
        try:
            with open(TREE_FILE_NAME, 'w') as f:
                json.dump(tree_json_data, f, indent=2)
            print(f"\n--- Game Tree Exported ---")
            print(f"Tree structure saved to {TREE_FILE_NAME}")
            print(f"Open 'tree_visualizer.html' and click 'Load Tree' to view the visualization.")
            print("--------------------------")
        except IOError as e:
            print(f"ERROR: Could not write tree JSON to file {TREE_FILE_NAME}. {e}")
            print("Printing to console as fallback:")
            print(json.dumps(tree_json_data, indent=2))
        
        # Print Stats
        print("--- AI Move Statistics ---")
        print(f"Time Taken: {self.working_time:.4f} seconds")
        print(f"Nodes Expanded: {self.nodes_expanded}")
        print(f"Max Depth Reached: {self.max_depth_reached} from start node")
        print(f"Nodes Pruned (Alpha-Beta): {self.pruned_nodes}")
        print(f"Best Score Found: {best_score}")
        print(f"Chosen Move (Column): {best_move_col}")

        return best_move_col

    def print_tree(self, node, depth=0):
        prefix = " " * depth * 2
        segments = [f"Move: {node.move}", f"Type: {node.node_type}", f"Value: {node.heuristic_value}"]
        if node.probability is not None and (node.node_type == "chance" or abs(node.probability - 1.0) > 1e-9):
            segments.append(f"Prob: {round(node.probability, 2)}")
        print(prefix + ", ".join(segments))
        for child in node.children:
            self.print_tree(child, depth + 1)

    def to_json_structure(self, node):
        """Recursively converts a Node structure to a serializable dictionary (JSON structure)."""
        if node is None:
            return {}

        node_type = getattr(node, "node_type", "decision")
        move = getattr(node, "move", -1)
        value = getattr(node, "heuristic_value", None)
        probability = getattr(node, "probability", 1.0)
        
        # Determine the player for Decision nodes
        player = None
        if node_type == "decision":
            depth = node.get_depth()
            
            # FIX: Use modulo 4 to correctly alternate MAX/MIN for Decision nodes 
            # in a structure that alternates Decision -> Chance -> Decision -> Chance
            if depth % 4 == 0:
                # Depth 0, 4, 8, ...
                player = "MAX (AI)"
            elif depth % 4 == 2:
                # Depth 2, 6, 10, ...
                player = "MIN (Human)"
            # Note: Depths 1, 3, 5... are Chance nodes, handled below.
            else:
                 # Fallback: Should not happen if game loop is clean, but treats 
                 # Minimax/Alpha-Beta (where depth 1 is MIN) as MIN.
                 player = "MIN (Human)" 
        
        # For chance nodes, the player is not a factor.
        elif node_type == "chance":
            player = "Chance"

        data = {
            "move": move,
            "type": node_type,
            "player": player,
            "value": round(value, 4) if value is not None else "N/A",
            "probability": round(probability, 4) if probability != 1.0 else 1.0,
        }
        
        children_list = []
        # Optimization: Limit recursion for JSON structure to prevent massive output file sizes
        # The limit is set to Depth 6.
        if node.get_depth() <= 6:
            for child in node.children:
                children_list.append(self.to_json_structure(child))
        
        data["children"] = children_list

        # Note if the branch was stopped for size reasons
        if node.get_depth() == 6 and len(node.children) > 0: 
            data["truncated"] = True
            
        return data
        
    # --- The rest of the AIAgent class methods are unchanged (omitted for brevity) ---

    def print_statistics(self):
        print(f"Nodes Expanded: {self.nodes_expanded}")
        print(f"Max Depth Reached: {self.max_depth_reached}")
        print(f"Pruned Nodes: {self.pruned_nodes}")
        print(f"Working Time: {self.working_time} seconds")


    def get_random_move(self, game):  
        # Get all possible moves
        valid_moves = game.get_valid_moves()
        
        # Return a random choice from the list
        if valid_moves:
            return random.choice(valid_moves)
        
        return None # Should not happen if game isn't over

    def _evaluate_state(self, game):
        # Choose between pure score evaluation and heuristic-based evaluation
        if self.heuristic_enabled:
            return self._evaluate_with_heuristic(game)
        return self._evaluate_score_only(game)

    def _evaluate_score_only(self, game):
        # Use the actual score difference (AI - Human)
        human_score, ai_score = game.count_connected_fours()
        return ai_score - human_score

    def _evaluate_with_heuristic(self, game):
        # Heuristic evaluation. Our heuristic is divided to:
        # 1. Completed connected-fours (weighted heavily)
        # 2. Potential threats (3-in-a-rows, 2-in-a-rows, 1-in-a-rows)
        # 3. Center / central columns occupation (slight preference for central pieces)
        # 4. Blocking opponent's threats (negative weight, symmetric to own threats)

        board = game.board
        ROWS, COLS = board.shape

        # Completed connected-fours (scaled by 1000 as requested)
        human_score, ai_score = game.count_connected_fours()
        score = (ai_score - human_score) * 1000

        # Build a column weight map: center columns get highest weight, borders get 1.
        col_weights = self._build_column_weights(COLS)

        # Build a rows weight map: similar to columns
        row_weights = self._build_column_weights(ROWS)

        # Add weight maps
        score += self._score_positional_map(board, game.AI, col_weights, row_weights)
        score -= self._score_positional_map(board, game.HUMAN, col_weights, row_weights)

        # We evaluate windows from AI perspective and subtract symmetric values for HUMAN.
        score += self._score_threats(board, game.AI)
        score -= self._score_threats(board, game.HUMAN)

        return score

    
    def _build_column_weights(self, cols: int):
        # Assign higher weights to central columns and weight 1 to border columns.
        # Central column(s) get weight = cols/2, weights decrease stepwise toward borders.

        
        if cols <= 0:
            return []
        max_weight = cols / 2.0 # Weight of the most central column(s)
        center = (cols - 1) / 2.0 # Center index (can be non-integer for even cols)
        weights = []
        for c in range(cols):
            distance = abs(c - center) # distance from center
            if max_weight <= 1:
                weights.append(1.0) # should not happen ever anyways
            else:
                # Linear drop: weight = max_weight - k * distance, clipped to [1, max_weight]
                # Choose k so that farthest column has weight ~= 1.
                max_distance = center # max distance from center to border
                if (max_distance == 0): # safety check just in case - should never happen
                    weights.append(1.0)
                    print("Warning: max_distance is zero in _build_column_weights.")
                    continue
                k = (max_weight - 1.0) / max_distance # slope
                w = max_weight - k * distance # this makes most central column weight = max_weight, borders = 1
                if w < 1.0:
                    w = 1.0
                    print("Warning: computed weight less than 1.0 in _build_column_weights.")
                weights.append(w)
        return weights

    def _build_row_weights(self, rows: int):
        # Assign higher weights to central rows and weight 1 to border rows.
        # Central row(s) get weight = rows/2, weights decrease stepwise toward borders.

        
        if rows <= 0:
            return []
        max_weight = rows / 2.0 # Weight of the most central row(s)
        center = (rows - 1) / 2.0 # Center index (can be non-integer for even rows)
        weights = []
        for r in range(rows):
            distance = abs(r - center) # distance from center
            if max_weight <= 1:
                weights.append(1.0) # should not happen ever anyways
            else:
                # Linear drop: weight = max_weight - k * distance, clipped to [1, max_weight]
                # Choose k so that farthest row has weight ~= 1.
                max_distance = center # max distance from center to border
                if (max_distance == 0): # safety check just in case - should never happen
                    weights.append(1.0)
                    print("Warning: max_distance is zero in _build_row_weights.")
                    continue
                k = (max_weight - 1.0) / max_distance # slope
                w = max_weight - k * distance # this makes most central row weight = max_weight, borders = 1
                if w < 1.0:
                    w = 1.0
                    print("Warning: computed weight less than 1.0 in _build_row_weights.")
                weights.append(w)
        return weights
    
    def _score_positional_map(self, board, piece, col_weights, row_weights):
        score = 0.0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == piece:
                    score += col_weights[c] * 0.5
                    score += row_weights[r] * 0.25 # rows are less important than columns
        return score
    
    def _score_threats(self, board, piece):
        # Scores all 4-cell windows based purely on piece count.

        score = 0
        ROWS, COLS = board.shape
        opponent = 1 if piece == 2 else 2 # opposite of player - aka piece

        # helper to score a single window
        def evaluate_window(window):
            count_piece = window.count(piece)
            count_empty = window.count(0) # empty tiles
            count_opp = window.count(opponent)

            # If window is blocked by opponent, it has 0 potential -> 0
            if count_opp > 0:
                return 0
            
            # Scoring weights
            if count_piece == 3 and count_empty == 1:
                return 100 # Strong Threat
            elif count_piece == 2 and count_empty == 2:
                return 5 # Minor Threat
            
            return 0

        # Horizontal
        for r in range(ROWS):
            row_array = list(board[r, :])
            for c in range(COLS - 3):
                score += evaluate_window(row_array[c:c+4])

        # Vertical
        for c in range(COLS):
            col_array = list(board[:, c])
            for r in range(ROWS - 3):
                score += evaluate_window(col_array[r:r+4])

        # Positive Slope Diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += evaluate_window(window)

        # Negative Slope Diagonal
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += evaluate_window(window)

        return score

    def _evaluate_expected_outcome(self, parent_node, game, selected_col, depth, is_ai_turn):
        # Build expectation node for the selected column and compute expected heuristic value
        selection_node = Node(self._copy_game_state(game), parent=parent_node, move=selected_col, node_type="chance")
        parent_node.add_child(selection_node)

        probabilities = self._get_slip_probabilities(game, selected_col)
        piece = game.AI if is_ai_turn else game.HUMAN

        expected_value = 0.0
        total_weight = 0.0

        for actual_col, probability in probabilities:
            if not game.is_valid_move(actual_col):
                continue
            next_state = self._copy_game_state(game)
            next_state.drop_piece(actual_col, piece)
            outcome_node = Node(next_state, parent=selection_node, move=actual_col, probability=probability)
            selection_node.add_child(outcome_node)
            
            # Call expectiminimax which correctly handles MAX/MIN based on the flag
            child_value = self.expectiminimax(outcome_node, depth - 1, not is_ai_turn)
            expected_value += probability * child_value
            total_weight += probability

        if total_weight == 0:
            selection_node.heuristic_value = self._evaluate_state(game)
            return selection_node.heuristic_value

        selection_node.heuristic_value = expected_value / total_weight
        return selection_node.heuristic_value

    def _get_slip_probabilities(self, game, selected_col):
        # Determine the probability distribution for where a piece actually lands
        probabilities: List[Tuple[int, float]] = []
        base_prob = 0.6 if game.is_valid_move(selected_col) else 0.0
        if base_prob > 0:
            probabilities.append((selected_col, base_prob))

        left_col = selected_col - 1
        right_col = selected_col + 1
        left_valid = left_col >= 0 and game.is_valid_move(left_col)
        right_valid = right_col < game.cols and game.is_valid_move(right_col)

        if left_valid and right_valid:
            probabilities.append((left_col, 0.2))
            probabilities.append((right_col, 0.2))
        elif left_valid:
            probabilities.append((left_col, 0.4))
        elif right_valid:
            probabilities.append((right_col, 0.4))

        total = sum(weight for _, weight in probabilities)
        if total == 0:
            return []

        return [(col, weight / total) for col, weight in probabilities]
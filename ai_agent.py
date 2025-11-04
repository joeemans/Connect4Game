import math
import random
import time

from game import Connect4Game

class Node:
    # Node class - data structure
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
       # self.alpha = float('-inf')
       # self.beta = float('inf')
        self.move = move # The move that led to this state
        self.heuristic_value = None

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
        self.reset_statistics() # Initialize statistics

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
        self.nodes_expanded += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
        game = node.state # Current game state

        if game.is_terminal_node() or depth == 0: 
            # Important: Evaluation at depth=0 considers current scores for unfinished games. This allows AI to make informed decisions even if it can't see a terminal state within the depth limit. This is in itself a heuristic evaluation but may not be sufficient for strong play. This needs further work.
            human_score, ai_score = game.count_connected_fours()
            node.heuristic_value = (ai_score - human_score) # Will use heuristic here to represent terminal state scores (difference in scores is the maximized value)
            return node.heuristic_value

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
 
    def get_move(self, game, depth_k, algorithm_type):
        # Entry point for AI to get its move
        print(f"\nAI is thinking (Algorithm: {algorithm_type}, Depth: {depth_k})...")
        
        # Reset stats and start timer
        self.reset_statistics()
        start_time = time.time()
        
        # Create the root node
        root_game_state = self._copy_game_state(game) # IMPORTANT: We are using a deep copy so the original game object is not changed
        root_node = Node(root_game_state, move=-1) # Root has no move leading to it

        if algorithm_type == "minimax":
            best_score = self.minimax(root_node, depth_k, True)
        elif algorithm_type == "alpha_beta":
            best_score = self.minimax_alpha_beta(root_node, depth_k, -math.inf, math.inf, True)
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

        # Print Stats
        print("--- AI Move Statistics ---")
        print(f"Time Taken: {self.working_time:.4f} seconds")
        print(f"Nodes Expanded: {self.nodes_expanded}")
        print(f"Max Depth Reached: {self.max_depth_reached} from start node")
        print(f"Nodes Pruned (Alpha-Beta): {self.pruned_nodes}")
        print(f"Best Score Found: {best_score}")
        print(f"Chosen Move (Column): {best_move_col}")

        # Print the Tree
        print("\n--- Game Tree ---")
        ###

        return best_move_col

    def print_tree(self, node, depth=0):
        # Print the game tree for debugging
        print(" " * depth * 2 + f"Move: {node.move}, Heuristic: {node.heuristic_value}")
        for child in node.children:
            self.print_tree(child, depth + 1)

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
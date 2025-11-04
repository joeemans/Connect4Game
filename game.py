# This is the logical module for a Connect 4 game.
# This module establishes the game board, validates moves, and detects wins.
# This is NOT the AI algorithm, or the GUI, but both will interact with this module.

import numpy as np


class Connect4Game:
    # This class manages the Connect 4 board and game logic.

    def __init__(self, rows=6, cols=7):
        if rows < 6 or cols < 7: # Enforce minimum board size as described in the pdf
            raise ValueError("Board must be at least 6 rows by 7 columns")
        self.rows = rows
        self.cols = cols
        self.EMPTY = 0 # Will represent empty cells with 0
        self.HUMAN = 1 # Will represent human player with 1
        self.AI = 2 # Will represent AI player with 2
        self.reset_board() # Initialize the board

    def reset_board(self): # Resets the board
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8) # Initialize the board as a 2D numpy array of zeros. Note: we can use int8 to save memory. Further optimization can be done if needed but it adds much complexity for the sake of this assignment.
        self.humanScore = 0
        self.aiScore = 0
        self.winner = None
        self.turn = self.HUMAN

    def is_valid_move(self, col): # Check if a move is valid (column not full)
        if col < 0 or col >= self.cols:
            return False
        return self.board[self.rows-1][col] == self.EMPTY # Top row empty means column is not full
    
    def get_valid_moves(self): # Get list of all valid column indices
        return [col for col in range(self.cols) if self.is_valid_move(col)]
    
    def get_next_open_row(self, col): # Get the next open row in a column
        for row in range(0, self.rows):
            if self.board[row][col] == self.EMPTY:
                return row
        return None # Column is full
    
    def drop_piece(self, col, piece): # Drop a piece into the board at the specified column
        row = self.get_next_open_row(col) # Find the next open row in the specified column

        if row is not None:
            self.board[row][col] = piece # Piece = 1 or 2 // Human or AI
            return True
        
        return False # Move was invalid (column full)
    
    def is_board_full(self): 
        # Check if the board is full
        return len(self.get_valid_moves()) == 0
    
    def is_terminal_node(self):
        # Check if the game has ended (win or draw)
        return self.is_board_full()

    def print_board(self):
        # Print the board to console top-down.
        print("\n" + "  ".join(str(i) for i in range(self.cols)))
        print("-" * (self.cols * 3))
        # Iterate from top row (rows-1) down to bottom row (0)
        for row in reversed(range(self.rows)):
            print(" ".join(["." if self.board[row][col] == self.EMPTY 
                            else "H" if self.board[row][col] == self.HUMAN 
                            else "A" for col in range(self.cols)]))
        print()

    # Now time to implement connected-four detection
    def count_sequence_at_index(self, row, col):
        # Count connected-fours starting from a specific index (row, col)

        if self.board[row][col] == self.EMPTY:
            return 0, self.EMPTY # Nothing to count from an empty cell. Should not happen anyways added just for completeness.

        piece = self.board[row][col]
        count = 0

        # Check horizontal (ends at [row, col])
        if col >= 3:
            horizontal_segment = self.board[row, col-3:col+1]
            if np.all(horizontal_segment == piece):
                count += 1
               # print("Found horizontal line at", row, col, piece)

        # Check vertical (ends at [row, col])
        if row >= 3:
            vertical_segment = self.board[row-3:row+1, col]
            if np.all(vertical_segment == piece):
                count += 1
               # print("Found | diagonal at", row, col, piece)

        # Check '\' diagonal - 
        if row <= self.rows - 4 and col >= 3:
            # Get the 4x4 sub-array ending at [row, col]
            sub_array = np.flipud(self.board[row:row+4, col-3:col+1])
            
            # Check '\' diagonal (top-left to bottom-right)
            if np.all(np.diag(sub_array) == piece):
                count += 1
               # print("Found \\ diagonal at", row, col, piece)
        
        if row >= 3 and col >= 3:
            # Get the 4x4 sub-array ending at [row, col]
            sub_array = np.flipud(self.board[row-3:row+1, col-3:col+1])
            
            # Check '/' diagonal (bottom-left to top-right)
            if np.all(np.diag(np.fliplr(sub_array)) == piece):
                count += 1
               # print("Found / diagonal at", row, col, piece)
        
        return count, piece
    
    def count_connected_fours(self):
        # Count all connected-fours for both players, and updates their scores
        self.humanScore = 0
        self.aiScore = 0
        for row in range(self.rows):
            for col in range(self.cols):
                count, piece = self.count_sequence_at_index(row, col)
                if piece == self.HUMAN:
                    self.humanScore += count
                elif piece == self.AI:
                    self.aiScore += count
    
        return self.humanScore, self.aiScore
    
    def determine_winner(self):
        # Determine the winner based on scores
        if self.humanScore > self.aiScore:
            self.winner = "PLAYER"
        elif self.aiScore > self.humanScore:
            self.winner = "AI"
        else:
            self.winner = "DRAW"
        return self.winner
    
    def paste_board(self, board: np.ndarray): # helper function for debugging
        if board.shape != (self.rows, self.cols):
            raise ValueError("Board shape does not match game dimensions")
        self.board = board.copy()


### TESTING

# Define constants for readability
EMPTY = 0
HUMAN = 1 # Red
AI = 2    # Yellow

# This is the board from the image, represented as a numpy array.
# Row 0 is the bottom-most row.
test_board = np.array([
    [HUMAN,    AI,    HUMAN, AI,    HUMAN,    AI,    AI],    # Row 0
    [AI,    AI,    HUMAN, HUMAN, AI,    HUMAN, AI],    # Row 1
    [HUMAN, HUMAN, HUMAN, HUMAN, AI, HUMAN, AI],    # Row 2
    [HUMAN, HUMAN, HUMAN, AI,    HUMAN,    AI, HUMAN],    # Row 3
    [HUMAN, HUMAN, AI,    AI,    AI,    HUMAN, AI],    # Row 4
    [HUMAN, AI,    HUMAN, AI,    AI,    AI, AI]     # Row 5 (Top)
], dtype=np.int8)
row = 2
col = 3
sub_array = np.flipud(test_board[row:row+4, col-3:col+1])
print(sub_array)

game = Connect4Game(6, 7)
game.paste_board(test_board)
humanScore, aiScore = game.count_connected_fours()
winner = game.determine_winner()
print("Test Board:")
game.print_board()
print(f"Human connected-fours: {humanScore}")
print(f"AI connected-fours: {aiScore}")
print(f"Winner: {winner}")
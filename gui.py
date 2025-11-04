import pygame
import sys
import math
from game import Connect4Game
from ai_agent import AIAgent

SQUARESIZE = 100 # Size of each square on the board
RADIUS = int(SQUARESIZE / 2 - 5)

# Colors to use
COLOR_BLUE = (0, 80, 200)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (220, 20, 60)
COLOR_YELLOW = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)

def get_game_dimensions():
    # Gets row and column input from the user in the console. Can be changed for GUI input later.
    while True:
        try:
            rows = input("Enter number of rows (default 6, min 6): ") or 6
            rows = int(rows)
            if rows < 6:
                print("Minimum rows is 6.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            cols = input("Enter number of columns (default 7, min 7): ") or 7
            cols = int(cols)
            if cols < 7:
                print("Minimum columns is 7.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
            
    return rows, cols

def draw_board(screen, board, ROWS, COLS, SQUARESIZE, RADIUS):
    
    # Fill the background first
    screen.fill(COLOR_BLACK)

    # Iterate over the VISUAL rows, from top to bottom
    for visual_row in range(ROWS):
        # The corresponding row in the numpy array (0 is bottom, ROWS-1 is top). They are inverted.
        numpy_row = (ROWS - 1) - visual_row
        
        for c in range(COLS):
            # Y-coordinate for the top of the blue rectangle
            # (visual_row + 1) accounts for the hover bar at the top
            rect_y = (visual_row + 1) * SQUARESIZE
            
            # Y-coordinate for the center of the circle
            circle_y = int(rect_y + SQUARESIZE / 2) 
            
            # X-coordinate is simpler
            rect_x = c * SQUARESIZE
            circle_x = int(rect_x + SQUARESIZE / 2)
            
            # Draw the blue rectangle
            pygame.draw.rect(screen, COLOR_BLUE, (rect_x, rect_y, SQUARESIZE, SQUARESIZE))

            # Get the piece from the correct numpy row
            piece = board[numpy_row][c]
            
            # Determine the color for the circle based on the piece
            draw_color = COLOR_BLACK # Default empty

            if piece == 1: # Human
                draw_color = COLOR_RED
            elif piece == 2: # AI
                draw_color = COLOR_YELLOW
            
            # Draw the circle
            pygame.draw.circle(screen, draw_color, (circle_x, circle_y), RADIUS)

    pygame.display.update()

def main():
    # Get dimensions from user before initializing pygame - possible to change later for GUI input
    ROWS, COLS = get_game_dimensions()
    
    # Calculate screen dimensions based on rows and columns
    WIDTH = COLS * SQUARESIZE
    HEIGHT = (ROWS + 1) * SQUARESIZE # extra row space for the top hover area

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4 Game")
    myfont = pygame.font.SysFont("monospace", int(SQUARESIZE * 0.75))

    game = Connect4Game(ROWS, COLS)
    ai = AIAgent()
    game_over = False
    
    draw_board(screen, game.board, ROWS, COLS, SQUARESIZE, RADIUS)
    pygame.display.update()

    while not game_over: # Game loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            # Dynamic circle hover effect
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, COLOR_BLACK, (0, 0, WIDTH, SQUARESIZE)) # Clear top bar
                posx = event.pos[0]
                if game.turn == game.HUMAN and not game_over: # Only show hover if not game over
                    pygame.draw.circle(screen, COLOR_RED, (posx, int(SQUARESIZE / 2)), RADIUS) # fixed up y-coordinate
                pygame.display.update()

            # Player click
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.turn == game.HUMAN and not game_over: # Check it is player's turn otherwise disregard
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    
                    if game.is_valid_move(col):
                        game.drop_piece(col, game.HUMAN)
                        game.turn = game.AI # Switch turn
                        draw_board(screen, game.board, ROWS, COLS, SQUARESIZE, RADIUS)
                    
                    if game.is_terminal_node():
                        game_over = True

        # AI move
        if game.turn == game.AI and not game_over:
            #col = ai.get_random_move(game)
            col = ai.get_move(game, depth_k=4, algorithm_type="minimax")
            
            if col is not None and game.is_valid_move(col):
                pygame.time.wait(500) 
                game.drop_piece(col, game.AI)
                game.turn = game.HUMAN # Switch turn
                draw_board(screen, game.board, ROWS, COLS, SQUARESIZE, RADIUS)
            
            if game.is_terminal_node():
                game_over = True

        # Game over handling
        if game_over:
            print("Game Over!")
            game.count_connected_fours()
            winner = game.determine_winner()
            
            print(f"Final Score: Player ({game.humanScore}) vs AI ({game.aiScore})")
            print(f"Winner: {winner}")

            pygame.draw.rect(screen, COLOR_BLACK, (0, 0, WIDTH, SQUARESIZE)) # Clear top bar
            if winner == "PLAYER":
                label = myfont.render("PLAYER WINS!", 1, COLOR_RED)
            elif winner == "AI":
                label = myfont.render("AI WINS!", 1, COLOR_YELLOW)
            else:
                label = myfont.render("IT'S A DRAW!", 1, COLOR_WHITE)
                
            # Center the label
            label_x = (WIDTH - label.get_width()) / 2
            label_y = (SQUARESIZE - label.get_height()) / 2
            screen.blit(label, (label_x, label_y))
            pygame.display.update()
            
            pygame.time.wait(10000)
            break # Exit the while loop

if __name__ == "__main__":
    main()
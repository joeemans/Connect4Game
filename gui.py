import pygame
import sys
import math
import time
import random

from game import Connect4Game
from ai_agent import AIAgent
from tree_visualizer import render_tree

SQUARESIZE = 100 # Size of each square on the board
RADIUS = int(SQUARESIZE / 2 - 5)

# Colors to use
COLOR_BLUE = (0, 80, 200)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (220, 20, 60)
COLOR_YELLOW = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)

ALGORITHMS = [
    ("Pure Minimax", "minimax"),
    ("Alpha-Beta", "alpha_beta"),
    ("Expectiminimax", "expectiminimax"),
]

class NumberInput:
    # helper for numeric text boxes on the start screen
    def __init__(self, label, default_value, min_value, position):
        self.label = label
        self.text = str(default_value)
        self.default_value = default_value
        self.min_value = min_value
        self.rect = pygame.Rect(position[0], position[1], 160, 40)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text += event.unicode

    def draw(self, screen, font):
        label_surface = font.render(self.label, True, COLOR_WHITE)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 28))
        display_text = self.text if self.text else ""
        text_surface = font.render(display_text, True, COLOR_WHITE)
        pygame.draw.rect(screen, COLOR_YELLOW if self.active else COLOR_WHITE, self.rect, 2)
        screen.blit(text_surface, (self.rect.x + 8, self.rect.y + 6))

    def get_value(self):
        if not self.text:
            return self.default_value
        try:
            value = int(self.text)
        except ValueError:
            value = self.default_value
        return max(value, self.min_value)

def draw_visualize_button(screen, rect, font, enabled=True):
    # Draws the visualize-tree button on the top bar
    fill_color = (70, 70, 70) if enabled else (40, 40, 40)
    border_color = COLOR_YELLOW if enabled else (120, 120, 120)
    pygame.draw.rect(screen, fill_color, rect, border_radius=6)
    pygame.draw.rect(screen, border_color, rect, 2, border_radius=6)
    label = "Visualize Tree" if enabled else "Tree Unavailable"
    text_surface = font.render(label, True, COLOR_WHITE)
    text_x = rect.x + (rect.width - text_surface.get_width()) / 2
    text_y = rect.y + (rect.height - text_surface.get_height()) / 2
    screen.blit(text_surface, (text_x, text_y))

def _calculate_slip_probabilities(game, selected_col):
    # Calculate the probabilities for slipping to adjacent columns
    probabilities = []

    if not game.is_valid_move(selected_col):
        return probabilities

    probabilities.append((selected_col, 0.6))

    left_col = selected_col - 1
    right_col = selected_col + 1
    left_valid = left_col >= 0 and game.is_valid_move(left_col)
    right_valid = right_col < game.cols and game.is_valid_move(right_col)

    if left_valid and right_valid: 
        probabilities.append((left_col, 0.2))
        probabilities.append((right_col, 0.2))
    elif left_valid:
        probabilities.append((left_col, 0.4)) # 0.4 if only one side is valid
    elif right_valid:
        probabilities.append((right_col, 0.4)) # same as above

    total = sum(weight for _, weight in probabilities)
    if total == 0:
        return []

    return [(col, weight / total) for col, weight in probabilities]

def sample_slipped_column(game, selected_col):
    # Sample the actual column based on slip probabilities
    probabilities = _calculate_slip_probabilities(game, selected_col)
    if not probabilities:
        return selected_col if game.is_valid_move(selected_col) else None
    cols, weights = zip(*probabilities)
    return random.choices(cols, weights=weights, k=1)[0]

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

def show_start_screen():
    # Display start screen for collecting rows, columns, depth, and algorithm selection
    WIDTH, HEIGHT = 1100, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4 - Setup")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("monospace", 42)
    body_font = pygame.font.SysFont("monospace", 24)

    row_input = NumberInput("Rows (>= 6)", 6, 6, (120, 320))
    col_input = NumberInput("Columns (>= 7)", 7, 7, (460, 320))
    depth_input = NumberInput("Depth K (>= 1)", 5, 1, (800, 320))
    inputs = [row_input, col_input, depth_input]

    algorithm_index = 0
    heuristic_enabled = False
    heuristic_button = pygame.Rect(640, 470, 390, 60)
    start_button = pygame.Rect(WIDTH/2 - 180, HEIGHT - 80, 360, 65)
    error_message = ""

    info_lines = [
        "Set the board size (rows ≥ 6, columns ≥ 7).",
        "Choose search depth K and algorithm.",
        "You can toggle heuristic evaluation.",
        "Expectiminimax introduces stochastic slips for both AI and player." 
    ]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            for input_box in inputs:
                input_box.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                algo_y = 400
                for idx, (label, _) in enumerate(ALGORITHMS):
                    rect = pygame.Rect(120, algo_y + idx * 75, 320, 60)
                    if rect.collidepoint(event.pos):
                        algorithm_index = idx
                if heuristic_button.collidepoint(event.pos):
                    heuristic_enabled = not heuristic_enabled
                elif start_button.collidepoint(event.pos):
                    rows = row_input.get_value()
                    cols = col_input.get_value()
                    depth_k = depth_input.get_value()

                    if rows < 6:
                        error_message = "Rows must be at least 6."
                    elif cols < 7:
                        error_message = "Columns must be at least 7."
                    elif depth_k < 1:
                        error_message = "Depth K must be 1 or more."
                    else:
                        _, algorithm_key = ALGORITHMS[algorithm_index]
                        return rows, cols, depth_k, algorithm_key, heuristic_enabled

        screen.fill(COLOR_BLACK)

        title_surface = title_font.render("Connect 4", True, COLOR_WHITE)
        screen.blit(title_surface, (WIDTH/2 - title_surface.get_width()/2, 60))

        for idx, line in enumerate(info_lines):
            info_surface = body_font.render(line, True, COLOR_WHITE)
            screen.blit(info_surface, (120, 170 + idx * 30))

        for input_box in inputs:
            input_box.draw(screen, body_font)

        algo_title = body_font.render("Select AI Algorithm", True, COLOR_WHITE)
        screen.blit(algo_title, (120, 370))

        for idx, (label, _) in enumerate(ALGORITHMS):
            rect = pygame.Rect(120, 400 + idx * 75, 320, 60)
            pygame.draw.rect(screen, COLOR_YELLOW if idx == algorithm_index else COLOR_WHITE, rect, 2)
            text_surface = body_font.render(label, True, COLOR_WHITE)
            screen.blit(text_surface, (rect.x + 16, rect.y + 18))

        heur_label = "Heuristic Evaluation: ON" if heuristic_enabled else "Heuristic Evaluation: OFF"
        pygame.draw.rect(screen, COLOR_YELLOW if heuristic_enabled else COLOR_WHITE, heuristic_button, 2)
        heur_text = body_font.render(heur_label, True, COLOR_WHITE)
        screen.blit(heur_text, (heuristic_button.x + 24, heuristic_button.y + 18))

        pygame.draw.rect(screen, COLOR_RED, start_button, border_radius=6)
        start_text = body_font.render("Start New Game", True, COLOR_WHITE)
        screen.blit(start_text, (start_button.x + (start_button.width - start_text.get_width())/2,
                                 start_button.y + (start_button.height - start_text.get_height())/2))

        if error_message:
            error_surface = body_font.render(error_message, True, COLOR_YELLOW)
            screen.blit(error_surface, (120, HEIGHT - 70))

        pygame.display.flip()
        clock.tick(60)

def main():
    pygame.init()
    config = show_start_screen()
    if config is None:
        return

    ROWS, COLS, depth_k, algorithm_key, heuristic_enabled = config

    # Calculate screen dimensions based on rows and columns
    WIDTH = COLS * SQUARESIZE
    HEIGHT = (ROWS + 1) * SQUARESIZE # extra row space for the top hover area

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4 Game")
    myfont = pygame.font.SysFont("monospace", int(SQUARESIZE * 0.75))
    button_font = pygame.font.SysFont("monospace", max(16, int(SQUARESIZE * 0.35)))
    viz_button_rect = pygame.Rect(WIDTH - 250, 10, 240, SQUARESIZE - 20)
    clock = pygame.time.Clock()

    game = Connect4Game(ROWS, COLS)
    ai = AIAgent()
    ai.set_heuristic_enabled(heuristic_enabled)
    game_over = False
    game_summary_displayed = False
    use_stochastic_physics = (algorithm_key == "expectiminimax")

    def refresh_visualize_button():
        # draw_visualize_button(screen, viz_button_rect, button_font, ai.get_last_tree() is not None)
        pygame.display.update(viz_button_rect)
    
    draw_board(screen, game.board, ROWS, COLS, SQUARESIZE, RADIUS)
    refresh_visualize_button()

    while True: # Game loop keeps running so the window stays open after the match
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Dynamic circle hover effect
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, COLOR_BLACK, (0, 0, WIDTH, SQUARESIZE)) # Clear top bar
                posx = event.pos[0]
                if game.turn == game.HUMAN and not game_over: # Only show hover if not game over
                    pygame.draw.circle(screen, COLOR_RED, (posx, int(SQUARESIZE / 2)), RADIUS) # fixed up y-coordinate
                refresh_visualize_button()
                pygame.display.update()

            # Player click
            if event.type == pygame.MOUSEBUTTONDOWN:
                if viz_button_rect.collidepoint(event.pos):
                    render_tree(ai.get_last_tree())
                    continue
                if game.turn == game.HUMAN and not game_over: # Check it is player's turn otherwise disregard
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    
                    if game.is_valid_move(col):
                        actual_col = col
                        if use_stochastic_physics:
                            slipped_col = sample_slipped_column(game, col)
                            if slipped_col is not None:
                                actual_col = slipped_col

                        if game.is_valid_move(actual_col):
                            game.drop_piece(actual_col, game.HUMAN)
                            game.turn = game.AI # Switch turn
                            draw_board(screen, game.board, ROWS, COLS, SQUARESIZE, RADIUS)
                            refresh_visualize_button()
                    
                    if game.is_terminal_node():
                        game_over = True

        # AI move
        if game.turn == game.AI and not game_over:
            col = ai.get_move(game, depth_k=depth_k, algorithm_type=algorithm_key)
            
            if col is not None and game.is_valid_move(col):
                pygame.time.wait(500) 
                actual_col = col
                if use_stochastic_physics:
                    slipped_col = sample_slipped_column(game, col)
                    if slipped_col is not None:
                        actual_col = slipped_col

                if game.is_valid_move(actual_col):
                    game.drop_piece(actual_col, game.AI)
                    game.turn = game.HUMAN # Switch turn
                    draw_board(screen, game.board, ROWS, COLS, SQUARESIZE, RADIUS)
                    refresh_visualize_button()
            
            if game.is_terminal_node():
                game_over = True

        # Game over handling
        if game_over and not game_summary_displayed:
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

            game_summary_displayed = True
            refresh_visualize_button()

        clock.tick(60) # Sleep to reduce CPU usage
        if game.turn == game.HUMAN and not game_over:
            time.sleep(0.01)

if __name__ == "__main__":
    main()
import React from "react";
import { useRef } from "react";

import { Anchor, Code } from "@mantine/core";
import { Prism } from "@mantine/prism";

import Eq from "../components/Eq";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleImage from "../components/ArticleImage";

import mnist from "../images/mnist.png";
import mnist_color from "../images/mnist_color.png";
import dropout_experiment_graph from "../images/dropout_experiment_graph.png";
import DropoutDiagram from "../images/DropoutDiagram.svg";

const TTTIntroPage = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Introduction",
      id: "introduction",
    },
    {
      name: "Writing the Game Logic",
      id: "game-logic",
    },
    {
      name: "Writing the UI",
      id: "ui",
    },
  ];

  function getHeaders() {
    return contentRef.current.querySelectorAll("h3");
  }

  return (
    <>
      <div className="article-wrapper">
        <ArticleNavigation
          sectionHeaders={sectionHeaders}
          getHeaders={getHeaders}
        />
        <div className="article-content-wrapper">
          <div className="article-content" ref={contentRef}>
            <ArticleTitle name={"Intro and Setup: RL for Games"} />
            <ArticleSubtitle name={"Let's Write a Tic-Tac-Toe Game"} />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              In 1997, IBM's Deep Blue, a supercomputer designed for the game of
              chess, defeated reigning world champion Garry Kaspasrov in a
              six-game match. Since the machine's victory, the gap between the
              performance of human chess players and that of chess engines has
              only grown. Today, computer analysis is an indispensible tool for
              many top players. When players prepare openings for tournament
              games, engine recommendations can help them uncover novel lines
              that they might not have otherwise considered.
            </p>
            <p>
              Considering the transformative impact computers had on the game of
              chess, you might be surprised that it wasn't until two decades
              after Kasparov's defeat that a computer was able to defeat a 9-dan
              professional Go player. DeepMind's AlphaGo managed to achieve this
              feat in 2016, when it won a five-game match against Lee Sedol. At
              this point, I owe you answers to at least a couple of questions:
            </p>
            <ol>
              <li>
                Chess and Go are ostensibly similar abstract strategy games, so
                why was it so much harder for a computer to achieve superhuman
                performance in Go?
              </li>
              <li>
                If Go is indeed fundamentally a much more difficult game than
                chess, what did AlphaGo do differently that rendered it capable
                of superhuman performance?
              </li>
            </ol>
            <p>
              To answer these questions, we'll need to start with something much
              simpler. Like chess and Go, tic-tac-toe is a game of perfect
              information, and it will be our toy example for learning about how
              to make AI for games. We'll be working with tic-tac-toe for
              several reasons:
              <ul>
                <li>
                  <strong>Simple to program: </strong>Tic-tac-toe can be
                  implemented easily in very few lines of code. We want to spend
                  as little time as possible worrying about the implementation
                  of the game, because our focus should be on the techniques
                  used to build a capable AI.
                </li>
                <li>
                  <strong>Simple to validate: </strong>It is more than feasible
                  for a human to fully evaluate any given tic-tac-toe position,
                  so it shouldn't be difficult for us to ensure our AI is doing
                  the right thing.
                </li>
                <li>
                  <strong>Non-trivial: </strong>Though tic-tac-toe can be fully
                  "solved" by a human, it certainly isn't trivial to write a
                  program capable of solving the game in a reasonable amount of
                  time. In other words, despite the simplicity of the game, it
                  will still be gratifying to see that our code works!
                </li>
                <li>
                  <strong>Scalable: </strong>We can easily scale the tic-tac-toe
                  board to dimensions larger than 3x3 to give our AI a more
                  complex task.
                </li>
                <li>
                  <strong>Similarities with Go: </strong>While tic-tac-toe is
                  far less complex than Go, the two games share many
                  elements--they are both turn-based games of perfect
                  information that are played on a grid--so it should be more or
                  less clear how our algorithms implemented for tic-tac-toe
                  could be adapted to Go.
                </li>
              </ul>
            </p>
            <p>
              In this post, we'll write a tic-tac-toe game in Python, including
              some simple graphics using pygame. The next post will discuss the
              minimax algorithm; with the alpha-beta pruning optimization, our
              code can fully evaluate a position in less than a second. By
              running our algorithm on a larger board, we'll uncover a need for
              new techniques which are designed not to fully solve a position,
              but instead provide progressively better estimates of the best
              move. The third post will discuss one such technique--Monte Carlo
              tree search--which is one component of DeepMind's AlphaGo. We'll
              then move on to discuss how AlphaGo combines Monte Carlo tree
              search with value networks and policy networks to achieve
              superhuman performance. Without further ado, let's dive in!
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              To develop the code for this article, I heavily modified{" "}
              <Anchor href="https://www.youtube.com/watch?v=KBpfB1qQx8w">
                this implementation
              </Anchor>{" "}
              written in pygame by Coding with Russ. I made the code more
              readable and concise in many places, and I neatly packaged
              different components of the code into classes to make it easier to
              extend. Since we'll be implementing our algorithms on top of this
              game, extensibility is important, but I tried to strike a balance
              between maintaining good object-oriented style and not ending up
              with an overengineered mess of classes.
            </p>
            <p>
              For our base tic-tac-toe game, we'll write three classes:{" "}
              <Code>Board</Code>, <Code>BoardDisplay</Code>, and{" "}
              <Code>MainDisplay</Code>. The <Code>Board</Code> class will manage
              the state of the game board, and will therefore perform functions
              such as resetting the game, determining if a move is legal, and
              checking if a player has won. The <Code>MainDisplay</Code> class
              will handle the pygame window and run the main game loop, and the{" "}
              <Code>BoardDisplay</Code> will render the board and handle events
              like clicking on a space. The display could easily be put in one
              class, but we're splitting it into two for a couple of reasons.
              First, we'll eventually want to be able to render boards other
              than the main board to display our algorithm's output of the best
              continuation of the game, so having a stand-alone class that can
              render a board will be useful. Second, having all of the display
              logic in a single class would mean that when we add in a
              visualization of the algorithm's evaluation, this class would get
              quite cluttered.
            </p>
            <p>
              Let's start by writing the <Code>Board</Code> class in a new file{" "}
              <Code>board.py</Code>. At the top of the file, we import numpy,
              which will make it simpler to implement some of our math
              operations. Then, we'll write a constructor that accepts a
              parameter <Code>board_dim</Code> to specify the number of rows and
              columns in the board, allowing us to scale our game. The
              constructor will set the <Code>board_dim</Code> instance variable
              and then make a call to <Code>reset_game()</Code>, which will set
              up a new game with a blank board.
            </p>
            <Prism language="python">{`import numpy as np

class Board:
    def __init__(self, board_dim):
        self.board_dim = board_dim
        self.reset_game()


    def reset_game(self):
        self.board_state = np.array([[0] * self.board_dim for _ in range(self.board_dim)])
        self.player = 1
        self.winner = None
        self.game_over = False`}</Prism>
            <p>
              To keep track of the board state, we'll use a 2D numpy array of
              shape of size <Code>(board_dim, board_dim)</Code>, where an entry
              of 0 represents an empty space, an entry of 1 represents an "X,"
              and an entry of -1 represents an "O". Here, we also initialize
              instance variables to keep track of the current player, the
              winner, and whether or not the game has ended.
            </p>
            <p>
              Next, we'll implement a method <Code>make_move()</Code>, which
              accepts a (row, column) tuple and places an "X" or "O" on the
              board at the specified position on the board. Each time we make a
              move, we'll need to check to see if a player has won the game, and
              we'll need to switch the current player. This will be accomplished
              by calls to <Code>update_winner()</Code> and{" "}
              <Code>switch_player()</Code>.
            </p>
            <Prism language="python">{`def make_move(self, move):
        cell_row, cell_col = move
        self.board_state[cell_row][cell_col] = self.player
        self.update_winner()
        self.switch_player()
        
def update_winner(self):
    row_sums = [sum(row) for row in self.board_state]
    col_sums = [sum(col) for col in self.board_state.T]
    diag_sums = [sum(diag) for diag in [np.diag(self.board_state), np.diag(np.fliplr(self.board_state))]]

    if self.board_dim in [*row_sums, *col_sums, *diag_sums]:
        self.winner = 1
        self.game_over = True
    elif -self.board_dim in [*row_sums, *col_sums, *diag_sums]:
        self.winner = 2
        self.game_over = True
    elif 0 not in self.board_state.flatten():
        self.winner = 0
        self.game_over = True
    else: 
        self.winner = None
        self.game_over = False
        
def switch_player(self):
    self.player *= -1
        `}</Prism>
            <p>
              To carry out the move, we unpack the tuple corresponding with the
              move and set the board state at this position to correspond with
              the current player (either 1 or -1). In{" "}
              <Code>update_winner()</Code>, we update the winner by using some
              nifty numpy operations to sum the rows, columns, and both
              diagonals of the board. If the sum of a row, column, or diagonal
              is equal to the number of elements <Code>board_dim</Code> in that
              line, it means that the line is filled with "X" markers, and thus
              "X" has won, and we set the winner to 1. Likewise, if the sum is
              equal to <Code>-board_dim</Code>, "O" has won, and we set the
              winner to 2. Finally, if the above conditions do not apply, and if
              there are no empty spaces remaining on the board, the game is a
              draw, and we set the winner to 0. In <Code>switch_player()</Code>,
              we simply multiply <Code>self.player</Code> by -1 to switch from 1
              to -1, and vice versa.
            </p>
            <p>
              Finally, we'll need to implement a few methods that will be used
              by the <Code>BoardDisplay</Code> class:{" "}
              <Code>move_is_legal()</Code> will return True if a move is both on
              the board and the space is unoccupied. We'll also include a few
              "getter" methods, <Code>get_winner()</Code>,{" "}
              <Code>get_game_over()</Code>, and <Code>get_board_state()</Code>{" "}
              to allow access to some instance variables.
            </p>
            <Prism language="python">{`def move_is_legal(self, move):
        (cell_row, cell_col) = move
        move_on_board = (0 <= cell_row < self.board_dim) and (0 <= cell_col < self.board_dim)
        if not move_on_board:
            return False
        return self.board_state[cell_row][cell_col] == 0
        
def get_winner(self):
    return self.winner


def get_game_over(self):
    return self.game_over


def get_board_state(self):
    return self.board_state
        `}</Prism>
            <p>
              Alright, that wraps up the game logic! Feel free to ensure that
              the code works by instantiating a Board object and calling some of
              the methods, like <Code>make_move()</Code>.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              Since the UI isn't really our focus here, I'll provide the code
              for the two classes, <Code>MainDisplay</Code> and{" "}
              <Code>BoardDisplay</Code>, with minimal explanation. For the{" "}
              <Code>MainDisplay</Code> class, create a new file{" "}
              <Code>main_display.py</Code>, and add the following code:
            </p>
            <Prism language="python">{`import pygame
from pygame.locals import *

from board_display import BoardDisplay

class MainDisplay:
    def __init__(self, board_dim, window_width):
        pygame.init()

        # Set up main window
        self.window_width = window_width

        self.bg_color = (15, 15, 15)

        self.screen = pygame.display.set_mode((self.window_width, self.window_width))
        self.screen.fill(self.bg_color)

        pygame.display.set_caption('Tic-Tac-Toe')

        # Set up main board display
        self.main_board_size = 5 * self.window_width // 6
        self.main_board_pos = (self.window_width // 12, self.window_width // 12)
        self.main_board_surface = pygame.Surface((self.main_board_size, self.main_board_size))
        self.main_board_display = BoardDisplay(self.main_board_surface, board_dim, self.main_board_size, True)


    def run_game(self):
        run = True
        while run:
            # Check for click
            clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONUP:
                    clicked = True        

            # Get mouse position
            pos = pygame.mouse.get_pos()
            pos = (pos[0] - self.main_board_pos[0], pos[1] - self.main_board_pos[1])
            
            # Draw main board
            self.main_board_display.draw_board(clicked, pos)
            self.screen.blit(self.main_board_surface, self.main_board_pos)

            pygame.display.update()

        pygame.quit()
    

if __name__ == "__main__":
    display = MainDisplay(board_dim=3, window_width=600)
    display.run_game()
        `}</Prism>
            <p>
              In the constructor, we create a window to render the board,
              setting the width and height of the window in pixels, the
              background color, and the window caption. We then create a surface
              in the center of the screen and instanstiate a{" "}
              <Code>MainBoardDisplay</Code> that will render the board onto this
              surface. In the <Code>run_game()</Code> method, we run the game
              loop, in which we check for click events. We pass arguments{" "}
              <Code>clicked</Code>, corresponding with whether or not a click
              has occurred, and
              <Code>pos</Code>, corresponding with the position of the mouse
              relative to the top-left of the game board, to the{" "}
              <Code>draw_board()</Code> method of the{" "}
              <Code>MainBoardDisplay</Code> object, which will draw onto{" "}
              <Code>self.main_board_surface</Code>. Finally, we use pygame's
              <Code>blit()</Code> method to draw the surface onto the screen.
            </p>
            <p>
              Finally, for the board display, create a new file called{" "}
              <Code>board_display.py</Code>, and add the following code:
            </p>
            <Prism language="python">
              {`import pygame
from pygame.locals import *

from board import Board


class BoardDisplay:
    def __init__(self, surface, board_dim, board_size, is_main_board, board=None):
        self.surface = surface
        self.board_dim = board_dim
        self.board_size = board_size
        self.is_main_board = is_main_board

        if board:
            self.board = board
        else:  
            self.board = Board(self.board_dim)

        self.font = pygame.font.SysFont(None, self.board_size // 15)

        self.bg_color = (30, 30, 30)
        self.msg_color = (0, 0, 0)
        self.msg_bg_color = (255, 255, 255)
        self.grid_color = (200, 200, 200)
        self.text_color = (255, 255, 255)
        self.x_color = (94, 96, 206)
        self.o_color = (72, 191, 227)

        self.line_width = self.board_size // 30
        self.grid_spacing = self.board_size // board_dim
        self.win_rect = Rect(self.board_size // 4, self.board_size // 4, self.board_size // 2, self.board_size // 8)
        self.again_rect = Rect(self.board_size // 3, self.board_size // 2, self.board_size // 3, self.board_size // 8)
    

    def draw_grid(self):
        for i in range(1, self.board_dim):
            # Draw horizontal line
            pygame.draw.line(self.surface, self.grid_color, (0, i * self.grid_spacing), (self.board_size, i * self.grid_spacing), self.line_width)
            # Draw vertical line
            pygame.draw.line(self.surface, self.grid_color, (i * self.grid_spacing, 0), (i * self.grid_spacing, self.board_size), self.line_width)


    def draw_x(self, cell_row, cell_col):
        padding = self.grid_spacing / 8

        cell_left = cell_col * self.grid_spacing + padding
        cell_right = (cell_col + 1) * self.grid_spacing - padding
        cell_top = cell_row * self.grid_spacing + padding
        cell_bottom = (cell_row + 1) * self.grid_spacing - padding

        pygame.draw.line(self.surface, self.x_color, (cell_left, cell_top), (cell_right, cell_bottom), self.line_width)
        pygame.draw.line(self.surface, self.x_color, (cell_right, cell_top), (cell_left, cell_bottom), self.line_width)


    def draw_o(self, cell_row, cell_col):
        padding = self.grid_spacing / 8
        radius = (self.grid_spacing / 2) - padding

        cell_center_x = cell_col * self.grid_spacing + self.grid_spacing / 2
        cell_center_y = cell_row * self.grid_spacing + self.grid_spacing / 2

        pygame.draw.circle(self.surface, self.o_color, (cell_center_x, cell_center_y), radius, self.line_width)


    def draw_markers(self):
        board_state = self.board.get_board_state()
        for i in range(self.board_dim):
            for j in range(self.board_dim):
                if board_state[i][j] == 1:
                    self.draw_x(i, j)
                if board_state[i][j] == -1:
                    self.draw_o(i, j)


    def draw_winner(self):
        if not self.board.get_game_over():  
            return

        winner = self.board.get_winner()

        win_text = "Draw!" if winner == 0 else f"Player {winner} wins!"
        win_img = self.font.render(win_text, True, self.msg_color)
        rect = win_img.get_rect()
        rect.center = self.win_rect.center
        pygame.draw.rect(self.surface, self.msg_bg_color, self.win_rect)
        self.surface.blit(win_img, rect)

        again_text = "Play again?"
        again_img = self.font.render(again_text, True, self.msg_color)
        rect = again_img.get_rect()
        rect.center = self.again_rect.center
        pygame.draw.rect(self.surface, self.msg_bg_color, self.again_rect)
        self.surface.blit(again_img, rect)


    def draw_board(self, clicked=False, pos=None):
        if self.is_main_board and clicked:
            if self.board.get_game_over():
                if self.again_rect.collidepoint(pos):
                    self.board.reset_game()
            else:
                cell_col = pos[0] // self.grid_spacing
                cell_row = pos[1] // self.grid_spacing
                if self.board.move_is_legal((cell_row, cell_col)):
                    self.board.make_move((cell_row, cell_col))

        self.surface.fill(self.bg_color)
        self.draw_grid()
        self.draw_markers()
        if self.is_main_board:
            self.draw_winner()
`}
            </Prism>
            <p>
              Running <Code>main_display.py</Code> should open up a window on
              which you can now play tic-tac-toe with a friend!
            </p>
            <img
              src="https://media.giphy.com/media/PioAZoey8gWNxE6Ha0/giphy.gif"
              alt="GIF of Playing Tic-Tac-Toe on a Board"
              width="50%"
            />
            <p>
              With this, we're ready to move on to the main event: writing AI to
              achieve perfect play.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default TTTIntroPage;

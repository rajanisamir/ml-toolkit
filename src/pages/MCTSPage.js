import React from "react";
import { useRef } from "react";

import { Anchor, Code } from "@mantine/core";
import { Prism } from "@mantine/prism";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleImage from "../components/ArticleImage";
import Eq from "../components/Eq";
import BlockEq from "../components/BlockEq";

import MCTSDiagram from "../images/MCTSDiagram.svg";
import MCTSSelectionDiagram1 from "../images/MCTSSelectionDiagram1.svg";
import MCTSSelectionDiagram2 from "../images/MCTSSelectionDiagram2.svg";

const MCTSPage = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Introduction",
      id: "introduction",
    },
    {
      name: "MCTS Fundamentals",
      id: "mcts-fundamentals",
    },
    {
      name: "Selecting Nodes",
      id: "selecting-nodes",
    },
    {
      name: "Why MCTS?",
      id: "why-mcts",
    },
    {
      name: "Implementing MCTS",
      id: "implementing-mcts",
    },
    {
      name: "Visualization Evaluations",
      id: "visualizing-evaluations",
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
            <ArticleTitle name={"Monte Carlo Tree Search"} />
            <ArticleSubtitle
              name={"Approximating Perfect Play in Complex Games"}
            />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              In the last article, we discussed and implemented the minimax
              algorithm, which was able to achieve perfect play on a 3x3
              tic-tac-toe board. By devising a lightweight optimization, known
              as alpha-beta pruning, we were able to significantly reduce the
              runtime of the algorithm, allowing it to fully evaluate the
              starting position in less than a second. Inspired by this success,
              we tried running the algorithm on a 4x4 board, but found that,
              even when given half an hour to run, the optimized algorithm did
              not terminate.
            </p>
            <p>
              The most straightforward way of dealing with this roadblock would
              be to reduce the depth of the minimax algorithm until it is able
              to run to completion in a reasonable amount of time. While this
              method could work in certain situations, it has a few major
              issues:
              <ul>
                <li>
                  <strong>Requires Static Evaluation: </strong> If we specify a
                  maximum depth for the minimax algorithm, there may be leaf
                  nodes containing boards in which the game has not yet ended.
                  Since leaf nodes must return a numerical value for each
                  position, we need to come up with some heuristic for
                  evaluating an arbitrary position on the board. For some games,
                  like tic-tac-toe, it isn't obvious how we would go about
                  producing such a static evaluation. Meanwhile, for games like
                  chess, there are obvious ways of producing a static
                  evaluation, like taking the difference between the total
                  values of each player's pieces. However, coming up with a
                  sophisticated method of evaluation that takes into account the
                  position on the board, rather than just each side's material,
                  proves almost impossible.
                </li>
                <li>
                  <strong>Wasted Computation Time: </strong>Suppose we allot a
                  certain amount of time to our minimax program to come up with
                  each move. Assuming we don't know <em>a priori</em> how long
                  running minimax will take to run at a given depth, we must run
                  the algorithm at some small initial depth, and then
                  progressively increase the depth until we run out of time.
                  Then, we'd report the move suggested by largest depth minimax
                  run that terminated. Since we're only using the move reported
                  by one run of minimax, each of the runs at a smaller depth, as
                  well as the largest-depth run, which would be in progress when
                  the allotted time expires, represent wasted computation time.
                  Ideally, we'd like our algorithm to continuously improve its
                  prediction as time goes on, rather than discretely as we
                  increase the depth.
                </li>
                <li>
                  <strong>Uniformly Limited Depth: </strong> If we cap the depth
                  of the minimax, each line of play explored by the algorithm
                  will be cut off at this same depth, prohibiting further
                  exploration. This cap on the depth means we may miss out on
                  some nuances of a position which become relevant only later
                  into the game. Ideally, we should be able to explore those
                  lines which look the most promising at a large depth, while
                  exploring lines which look comparatively weak at a small
                  depth.
                </li>
              </ul>
              These issues might be summarized as follows: "minimax is suited to
              produce perfect solutions for low-complexity games, but we want
              approximate solutions for high-complexity games." In this article,
              we'll be looking at Monte Carlo tree search (MCTS), which deals
              with all three of these issues. MCTS is used as the foundation of
              the architecture for AlphaGo, and it was used in all of the
              strongest Go programs before AlphaGo's breakthrough. After a
              theoretical discussion of the algorithm, we'll implement it on top
              of our tic-tac-toe game and test it out on a 4x4 board.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              I'll first describe how MCTS works, and then we'll address how it
              solves the three major issues we identified with using a
              depth-limited minimax algorithm. Monte Carlo tree search maintains
              a game tree, and it proceeds by repeating a series of four steps
              until the allotted time expires:
              <ol>
                <li>
                  <strong>Selection: </strong>Choose a leaf node from the tree.
                  In this context, "leaf node" refers both to nodes with no
                  children, and to nodes from which there exists a legal move
                  that does not have a corresponding node in the tree. In the
                  next section, we will say more about how leaf nodes are
                  selected; this selection process is critical to the
                  performance of MCTS.
                </li>
                <li>
                  <strong>Expansion: </strong>Choose uniformly at random a legal
                  move from the leaf node that has not yet been explored, and
                  create a node in the tree corresponding to the board state
                  after this move, initializing it with a visit count of 0 and a
                  score of 0.
                </li>
                <li>
                  <strong>Simulation: </strong>Perform a "rollout" from the
                  newly created node by selecting legal moves uniformly at
                  random for each player until the game has ended.
                </li>
                <li>
                  <strong>Backpropagation: </strong>At each node, increment the
                  number of visits by 1. Additionally, if the player who made
                  the most recent move won the game, increment the node's score
                  by 1 (if the game was a tie, increment the score by 0.5).
                </li>
              </ol>
              The diagram below shows what one iteration of MCTS might look like
              on a game tree.
            </p>
            <ArticleImage
              src={MCTSDiagram}
              width="90%"
              caption="One iteration of the MCTS algorithm on a game tree. In the selection phase, a leaf node is chosen. Then, in the expansion phase, a legal move is chosen from the board state at the leaf node, and a new node is created corresponding with the new board state. In the simulation phase, a game is played to completion using random rollouts, and in the backpropagation phase, the values of each of the nodes on the path from the root to the newly created node are updated. These values are represented by a fraction, where the numerator corresponds with the number of wins, and the denominator corresponds with the number of visitations."
            />
            <p>
              When the time allotted to select a move has passed, MCTS selects
              the move from the root node with the largest value (i.e. the
              highest win ratio).
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              We have not yet specified how a leaf node should be selected in
              the selection phase. To decide how we should go about this, let's
              first consider what might happen if we take a naive approach:
              starting at the root and select moves uniformly at random until we
              arrive at a leaf node. Suppose the game tree looks as follows:
            </p>
            <ArticleImage src={MCTSSelectionDiagram1} width="60%" />
            <p>
              The move labelled "wins for blue" will always lead to a win for
              the blue player, and the move labelled "loses for blue" will
              always lead to a win for the purple player. If the selection phase
              proceeds by selecting moves uniformly at random from the root,
              each time node A is reached, there is a 50% chance the move that
              wins for blue will be chosen, and there is a 50% chance the move
              that loses for blue will be chosen. Therefore, an equal number of
              wins and losses will be backpropagated to node A, and its win
              percentage will converge to 50%. However, under perfect play, the
              win percentage for node [A] should be 0%, because at node [A],
              purple made the most recent move, but blue can guarantee a win by
              choosing the move on the left.
            </p>
            <p>
              So, what went wrong here? By selecting moves at random during the
              selection phase, we did not take into account that, under perfect
              play, the current player will choose the move with the highest win
              percentage. This caused our position evaluations to converge to
              the wrong values. Therefore, instead of selecting moves at random,
              we should bias our choices in the selection phase toward those
              moves which lead to the highest win percentage for the current
              player. Not only will this provide us with more accurate position
              evaluations, but it will also lead to an increase in the
              exploration of the most optimal lines.
            </p>
            <p>
              While it might be tempting to <em>always</em> choose the optimal
              moves during the selection phase, this strategy also wouldn't
              quite work out. To demonstrate how this strategy might fail,
              consider the following game tree early on in MCTS:
            </p>
            <ArticleImage src={MCTSSelectionDiagram2} width="40%" />
            <p>
              Two iterations of MCTS have been carried out, and one new node was
              created in each iteration. The random rollout from the left node
              led to a loss for purple, and the random rollout from the right
              node led to a win for purple. Now, since the win percentage for
              the left node is 0%, and the win percentage for the right node
              will always be larger than 0%, choosing the optimal move from the
              root means always selecting the move on the right. Even though it
              could easily be possible that the left move is far better than the
              right move, we will never again explore the move on the left.
            </p>
            <p>
              Okay, what went wrong this time? Well, we didn't have much
              information about either of the moves available from the root, but
              by always selecting the optimal move during the selection phase,
              we implicitly assumed that we knew the exact evaluations of both
              children. The solution to this issue is to bias our moves during
              the selection phase toward the optimal moves for each player, but
              to have this bias depend on how much information we know about the
              available moves. The generalization of this tradeoff between the{" "}
              <em>exploration</em> of moves for which we have little
              information, and the <em>exploitation</em> of the information we
              have gathered so far, is known as the "multi-armed bandit
              problem," and I would encourage you to read more about the problem
              and the many situations in which it shows up.
            </p>
            <p>
              The good news for us is that the exploration-exploitation tradeoff
              has already been, in some sense, "solved" for MCTS. In other
              words, we know a strategy for the selection of nodes that
              guarantees MCTS will eventually converge to perfect play. This
              strategy, known as UCT (Upper Confidence bound for Trees) was
              designed by{" "}
              <Anchor
                href="http://ggp.stanford.edu/readings/uct.pdf"
                target="_blank"
              >
                Levente Kocsis and Csaba Szepesvári
              </Anchor>
              , and it is based off the UCB1 (Upper Confidence Bound 1) strategy
              devised by{" "}
              <Anchor
                href="https://homes.di.unimi.it/%7Ecesabian/Pubblicazioni/ml-02.pdf"
                target="_blank"
              >
                Auer et al.
              </Anchor>{" "}
              in an analysis of the multi-armed bandit problem. If you are
              interested in a theoretical proof of the strategy's viability, I'd
              recommend checking out the paper by Auer et al. for a more
              detailed treatment.
            </p>
            <p>
              The UCT strategy proceeds as follows: during the selection phase,
              to choose the next child node, compute a UCT score for each child
              node, and select the child node with the largest score. The UCT
              score is computed as:
            </p>
            <BlockEq
              text="$UCT = \frac{s}{n} + c\sqrt{\frac{ln(N)}{n}}$"
              displayMode={true}
            />
            <p>
              where <Eq text="$s$" /> is the score of a node (equal to the
              number of wins plus half the number of draws)
              <Eq text="$n$" /> is the number of times a node has been visited,{" "}
              <Eq text="$N$" /> is the number of times a node's parent has been
              visited, and <Eq text="$c$" /> is a tunable hyperparameter. The
              first term, which is higher for nodes with better winning
              percentages, can be thought of as the "exploitation" term, while
              the second term can be thought of as the "exploration" term. Thus,{" "}
              <Eq text="$c$" /> can be thought of as controlling the amount of
              exploration to perform during the selection phase. When we
              implement MCTS, I will use <Eq text="$c = \sqrt{2}$" />, the
              theoretical value used in the paper by Auer et al.
            </p>
            <p>
              While I won't mathematically justify the exact form of exploration
              term here, it's fairly easy to understand intuitively why the term
              encourages exploration. When the number of times <Eq text="$n$" />{" "}
              a node has been visited is small, the exploration term is large.
              Thus, nodes for which we do not have much information will be
              visited more frequently. As we become more confident about our
              estimate for the value of a node, the exploration term shrinks,
              because exploring that node again doesn't give us as much new
              information. The exploration term also increases as the number of
              times <Eq text="$N$" /> a node's parent has been visited grows.
              This can be thought of as providing a counterbalance to the
              shrinkage of the exploration term by ensuring that the rate of
              exploration doesn't grow too small, too quickly. Together, these
              terms ensure that we explore enough that we learn about lines of
              play that are difficult to find, but not so much that the
              evaluations of positions get thrown off.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
            <p>
              Now that we've covered how MCTS works, let's revisit our issues
              with the depth-limited minimax algorithm to understand why MCTS
              might be a better choice.
              <ul>
                <li>
                  <strong>Requires Static Evaluation: </strong> While the
                  depth-limited minimax algorithm required devising a heuristic
                  to produce a static evaluation of positions, MCTS simulates
                  games to completion and thus does not require a static
                  evaluation.
                </li>
                <li>
                  <strong>Wasted Computation Time: </strong>The depth-limited
                  minimax algorithm wastes computation time, because it must at
                  multiple depths, only one of which will be used to produce the
                  final move. By contrast, MCTS continuously improves its
                  predictions as time goes on.
                </li>
                <li>
                  <strong>Uniformly Limited Depth: </strong>The depth-limited
                  minimax algorithm limits the exploration of all lines of play
                  to a specified depth, which could cause issues if a large
                  depth is necessary to fully comprehend a complex line of play.
                  Instead, the depth to which MCTS explores in the selection
                  phase is not explicitly bounded. As nodes are added to the
                  game tree in the expansion phase, more of the game is played
                  out in the selection phase and less of the game is played out
                  in the simulation phase, where moves are selected at random.
                  In the selection phase, we can select moves according to a
                  policy, so that optimal lines of play will be explored more
                  often than suboptimal ones. This leads to more expansion
                  steps, and thus a larger effective search depth, in the most
                  promising lines.
                </li>
              </ul>
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[4]} />
            <p>
              Now that we understand why MCTS can alleviate many of the issues
              with the minimax algorithm, let's implement it on our tic-tac-toe
              game in Python to see how it works in practice. Create a new file
              called <Code>mcts.py</Code> with the following imports:
            </p>
            <Prism language="python">
              {`import numpy as np
import math
import copy
import time
from board import Board`}
            </Prism>
            <p>
              At the top of the file, we'll also define a constant{" "}
              <Code>c</Code> for the UCT exploration coefficient.
            </p>
            <Prism language="python">{`c = math.sqrt(2)`}</Prism>
            <p>
              We will define a class <Code>MCTSNode</Code>, corresponding with
              one node in the game tree. This class will support methods for the
              selection, expansion, simulation, and backpropagation phases of
              MCTS, each of which will be initiated from the node on which they
              are called. First, let's define the constructor for the{" "}
              <Code>MCTSNode</Code> class.
            </p>
            <Prism language="python">
              {`class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent  

        self.children = []
        self.remaining_moves = self.board.get_legal_moves()
        self.score = 0
        self.visits = 0`}
            </Prism>
            <p>
              The constructor accepts a parameter <Code>board</Code> that
              represents the state of the board at the node, as well as a
              parameter <Code>parent</Code> corresponding with the node's parent
              in the game tree. The node will also maintain a list of its
              children, a list of the remaining moves available from the node
              which do not yet have corresponding nodes in the game tree, its
              score, and the number of times it has been visited.
            </p>
            <p>
              Next, we'll implement the selection phase, which will require
              adding three additional methods to the <Code>MCTSNode</Code>{" "}
              class. To select a leaf node, we'll start at the root, and
              continue selecting children (via their UCT scores) until either
              the game is over or the current node is not fully expanded (i.e.
              there is at least one legal move without a corresponding node in
              the tree). We'll implement methods{" "}
              <Code>is_fully_expanded()</Code> and <Code>get_game_over()</Code>{" "}
              to check whether or not we should continue selecting children, and
              a method <Code>get_uct()</Code> to determine which child we should
              select. To compute the UCT score in <Code>get_uct()</Code>, we'll
              need to be able to get the number of visits of a node's parent, so
              we'll also implement a getter method to return{" "}
              <Code>self.visits</Code>.
            </p>
            <Prism language="python">
              {`def select(self):
    curr_node = self
    while curr_node.is_fully_expanded() and not curr_node.get_game_over():     
        ucts = [child.get_uct() for child in curr_node.children]
        curr_node = curr_node.children[np.argmax(ucts)]
    return curr_node
    

def is_fully_expanded(self):
    return len(self.remaining_moves) == 0


def get_game_over(self):
    return self.board.get_game_over()

    
def get_uct(self):
    return self.score / self.visits + c * math.sqrt(math.log(self.parent.get_visits()) / self.visits)


def get_visits(self):
    return self.visits
    `}
            </Prism>
            <p>
              To implement the expansion method, we'll first check if game has
              ended at the current node; if so, no new node can be created, and
              we'll just return the current node. Otherwise, we'll pop one legal
              move from the list of remaining unexplored moves, make this move
              on the board, and create a child node corresponding with the new
              board state after making the move.
            </p>
            <Prism language="python">
              {`def expand(self):
    if self.board.get_game_over():
        return self
    move = self.remaining_moves.pop()
    board = copy.deepcopy(self.board)
    board.make_move(move)
    child = MCTSNode(board, self)
    self.children.append(child)
    return child
    `}
            </Prism>
            <p>
              For the simulation phase, we'll continue making moves, chosen
              uniformly at random, from the current board state, until the game
              has ended. Finally, we'll return the winner of the game.
            </p>
            <Prism language="python">
              {`def simulate(self):
    board = copy.deepcopy(self.board)
    while not board.get_game_over():
        legal_moves = board.get_legal_moves()
        move = legal_moves[np.random.choice(len(legal_moves))]
        board.make_move(move)
    winner = board.get_winner()
    return winner`}
            </Prism>
            <p>
              Finally, the backpropagation phase is implemented by a method that
              accepts a parameter that specifies the winner of the simulation.
              The method will walk up the tree, updating the score and number of
              visits for each node, depending on the winner of the game. If the
              winner is the opposite of the player whose turn it is at a given
              node, it means the player who made the last move won the game, and
              we should increase the score of that node by 1. If the game is a
              draw, we'll increase the score of the node by 0.5. We'll update
              the score and number of visits for a node in a method{" "}
              <Code>update_value()</Code>, and we'll implement a method{" "}
              <Code>get_parent()</Code> that allows us to traverse up the tree.
            </p>
            <Prism language="python">
              {`def backpropagate(self, winner):
    curr_node = self
    while curr_node:
        curr_node.update_value(winner)
        curr_node = curr_node.get_parent() 
          
        
def update_value(self, winner):
    if (self.board.player == 1 and winner == 2) or (self.board.player == -1 and winner == 1):
        self.score += 1
    elif winner == 0:
        self.score += 0.5
    self.visits += 1

def get_parent(self):
    return self.parent
    `}
            </Prism>
            <p>
              Finally, we'll write a method to be invoked on the root of the
              game tree, which will perform each of the four steps of MCTS for a
              specified number of seconds, and then return the best move from
              the root. Instead of only returning the best move, we'll return a
              dictionary of each legal move from the root, along with its
              evaluation, which will allow us to visualize the recommendations
              of MCTS on the game board. To do this, we'll implement a couple of
              other methods: <Code>get_value()</Code>, which will return a
              node's win radio, and <Code>get_last_move()</Code>, to get the
              last move that was made to arrive at a node.
            </p>
            <Prism language="python">
              {`def get_move_evaluations(self, time_allocation):
    end_time = time.time() + time_allocation
    while time.time() < end_time:
        leaf = self.select()
        child = leaf.expand()
        winner = child.simulate()
        child.backpropagate(winner)
    evaluations = {child.get_last_move(): child.get_value() for child in self.children}
    return evaluations
    

def get_value(self):
    return self.score / self.visits


def get_last_move(self):
    return self.board.get_move_history()[-1]   
    `}
            </Prism>
            <p>
              That's it! To make sure everything is working, we can invoke MCTS
              on the starting position of a 3x3 board for something like 15
              seconds.
            </p>
            <Prism language="python">
              {`if __name__ == "__main__":
    board = Board(3)

    root = MCTSNode(board)
    evaluations = root.get_move_evaluations(time_allocation=15)
    print(evaluations)`}
            </Prism>
            <p>
              If all went well, the output should be a dictionary containing the
              win ratios of each move, which will likely be somewhere around
              0.5.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[5]} />
            <p>
              Finally, we'll write some code to visualize the evaluations of
              each move. Rather than showing the evaluation of the position, as
              we did for the minimax algorithm, we'll display the win ratio of
              each move in the Monte Carlo rollouts directly on the board. We'll
              work with the code we ended with last time, beginning with{" "}
              <Code>main_display.py</Code>. We'll go ahead and remove the{" "}
              <Code>EvaluationDisplay</Code> import and modify the constructor
              to set up a square window, with just the board in the center.
            </p>
            <Prism language="python">
              {`def __init__(self, board_dim, window_width):
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
            `}
            </Prism>
            <p>
              In the <Code>run_game()</Code> function, we'll remove the code
              that retrieves and draws the latest evaluation; this will be done
              directly in the <Code>BoardDisplay.draw_board()</Code> function.
              Also, before the call to <Code>pygame.quit()</Code>, we'll invoke
              on the board display a method called{" "}
              <Code>stop_evaluation()</Code>, which we'll implement soon, to
              tell our MCTS thread to stop running. The <Code>run_game()</Code>{" "}
              function should now look something like this:
            </p>
            <Prism language="python">
              {`def run_game(self):
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
    
    self.main_board_display.stop_evaluation()
    pygame.quit()`}
            </Prism>
            <p>
              Finally, we'll need to modify the <Code>BoardDisplay</Code> class
              to accomodate our new evaluation. At the top of the file, we can
              remove the import statement for the <Code>math</Code> module,
              which we'll no longer need, and we can import the{" "}
              <Code>MCTSNode</Code> class instead of the{" "}
              <Code>pruned_minimax()</Code> function:
            </p>
            <Prism language="python">
              {`import pygame
from pygame.locals import *
import copy
from threading import Thread

from board import Board
from mcts import MCTSNode`}
            </Prism>
            <p>
              To display the evaluations on the game board, we'll display a
              green square on each space corresponding with a legal move, along
              with text for the win ratio for that move. We'll use darker
              squares to denote weaker moves and lighter squares to denote
              stronger moves. Therefore, it makes sense to use the HSV
              (hue/saturation/value) color system instead of RGB, because we can
              vary the "value" to adjust the darkness of the square. To this
              end, in the constructor for the <Code>BoardDisplay</Code>, we'll
              initialize instance variables <Code>eval_hue</Code> and{" "}
              <Code>eval_saturation</Code> for the hue and saturation of the
              squares. In the constructor, we'll also initialize an instance
              variable <Code>self.run</Code> to <Code>True</Code>, which will
              keep track of whether the game is still running. This will allow
              the evaluation thread to know when to stop running Monte Carlo
              rollouts.
            </p>
            <Prism language="python">
              {`def __init__(self, surface, board_dim, board_size, is_main_board, board=None):
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
    self.eval_hue = 127
    self.eval_saturation = 88

    self.line_width = self.board_size // 30
    self.grid_spacing = self.board_size // board_dim
    self.win_rect = Rect(self.board_size // 4, self.board_size // 4, self.board_size // 2, self.board_size // 8)
    self.again_rect = Rect(self.board_size // 3, self.board_size // 2, self.board_size // 3, self.board_size // 8)
    
    if self.is_main_board:
        self.run = True
        self.update_evaluation()`}
            </Prism>
            <p>
              In <Code>update_evaluation()</Code>, we'll set{" "}
              <Code>self.evaluations</Code> to an empty dictionary, and we'll
              initialize the root node of the game tree with a copy of the
              current game board. Then, as before, we'll start a thread to
              execute the <Code>run_evaluation()</Code> method, which will
              accept a parameter for the root of the game tree.
            </p>
            <Prism language="python">
              {`def update_evaluation(self):
    self.evaluations = {}
    self.eval_root = MCTSNode(copy.deepcopy(self.board))
    t = Thread(target=self.run_evaluation, args=(self.eval_root,))
    t.start()`}
            </Prism>
            <p>
              In the <Code>run_evaluation</Code> method, we'll continue running
              MCTS rollouts in 0.1-second bursts. Before updating{" "}
              <Code>self.evaluations</Code> with the latest estimates, we'll
              make sure the root of the game tree hasn't changed (i.e. a new
              move has been made) and <Code>self.run</Code> is set to{" "}
              <Code>True</Code> (i.e. the game hasn't been quit). If a new move
              has been made or the game has been quit, we'll break from the
              evaluation.
            </p>
            <p>
              We'll also implement the <Code>stop_evaluation()</Code> function,
              which is called before quitting the game, to set{" "}
              <Code>self.run</Code> to <Code>False</Code>, and we'll modify the{" "}
              <Code>get_evaluation()</Code> function to return just the{" "}
              <Code>evaluation</Code> dictionary:
            </p>
            <Prism language="python">
              {`def stop_evaluation(self):
    self.run = False
    

def get_evaluation(self):
    return self.evaluations`}
            </Prism>
            <p>
              Finally, we'll implement the <Code>draw_evaluations()</Code>{" "}
              function to render the evaluation squares onto the board, using
              each move's evaluation as the corresponding square's brightness,
              and we'll call this function in <Code>draw_board()</Code>.
            </p>
            <Prism language="python">
              {`def draw_evaluations(self):
    padding = self.grid_spacing / 8
    for move, evaluation in self.evaluations.items():
        cell_row, cell_col = move
        rect_left = cell_col * self.grid_spacing + padding + self.line_width / 2
        rect_top = cell_row * self.grid_spacing + padding + self.line_width / 2
        rect_width = self.grid_spacing - self.line_width - padding * 2 
        rect_height = self.grid_spacing - self.line_width - padding * 2 
        eval_rect = Rect(rect_left, rect_top, rect_width, rect_height)

        color = pygame.Color(0)
        color.hsva = (self.eval_hue, self.eval_saturation, evaluation * 100, 1)

        pygame.draw.rect(self.surface, color, eval_rect)

        eval_text = f"{evaluation:.2f}"
        eval_img = self.font.render(eval_text, True, self.text_color)
        rect = eval_img.get_rect()
        rect.center = eval_rect.center

        self.surface.blit(eval_img, rect)
        
        
def draw_board(self, clicked=False, pos=None):
    if self.is_main_board and clicked:
        if self.board.get_game_over():
            if self.again_rect.collidepoint(pos):
                self.board.reset_game()
                self.update_evaluation()
        else:
            cell_col = pos[0] // self.grid_spacing
            cell_row = pos[1] // self.grid_spacing
            if self.board.move_is_legal((cell_row, cell_col)):
                self.board.make_move((cell_row, cell_col))
                self.update_evaluation()

    self.surface.fill(self.bg_color)
    self.draw_grid()
    self.draw_markers()
    if self.is_main_board:
        self.draw_winner()
        self.draw_evaluations()
        `}
            </Prism>
            <p>
              If all has gone well, running <Code>main_display.py</Code> on the
              board should allow you to visualize the estimations of MCTS. Below
              is an example of the visualization running on my machine, on a 4x4
              board.
            </p>
            <img
              src="https://media.giphy.com/media/9pFKwEYvrRjpXoNgcB/giphy.gif"
              alt="GIF of Playing Tic-Tac-Toe on a Board, with MCTS Evaluations on Each Square"
              width="50%"
            />
          </div>
        </div>
      </div>
    </>
  );
};

export default MCTSPage;

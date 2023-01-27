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

import TTTGameTree from "../images/TTTGameTree.svg";
import ArbitraryGameTree from "../images/ArbitraryGameTree.svg";
import ArbitraryGameTreeAnimation from "../images/ArbitraryGameTreeAnimation.webp";
import AlphaBetaPruningTree from "../images/AlphaBetaPruningTree.svg";
import AlphaBetaPruningTreePruned from "../images/AlphaBetaPruningTreePruned.svg";

const MinimaxPage = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Introduction",
      id: "introduction",
    },
    {
      name: "Understanding the Minimax Algorithm",
      id: "understanding-minimax",
    },
    {
      name: "Implementing Minimax",
      id: "implementing-minimax",
    },
    {
      name: "Alpha-Beta Pruning",
      id: "pruning",
    },
    {
      name: "Implementing Pruning",
      id: "pruning",
    },
    {
      name: "Beyond Exact Solutions",
      id: "beyond-exact-solutions",
    },
    {
      name: "Visualizing Our Evaluation",
      id: "visualizing",
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
            <ArticleTitle name={"Minimax and Alpha-Beta Pruning"} />
            <ArticleSubtitle name={"Achieving Perfect Play in Tic-Tac-Toe"} />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              Tic-tac-toe is a game of perfect information: at any point in the
              game, each player is aware of every action that has affected the
              state of the board. However, when a player makes a move, they
              cannot be certain about how their opponent will respond. Thus, in
              order to achieve perfect play, a player must evaluate each
              candidate move under the assumption that their opponent will
              respond optimally. This insight is at the heart of the minimax
              algorithm, which, given enough time, achieves perfect play by
              finding the move that <em>minimizes</em> the <em>maximium</em>{" "}
              value the opponent could potentially achieve.
            </p>
            <p>
              In this article, we'll learn about how the minimax algorithm works
              and discuss a lightweight optimization known as alpha-beta
              pruning, which can drastically improve its performance. We'll
              implement both the algorithm and its optimization on our
              tic-tac-toe program. Finally, we'll discuss why minimax can be
              impractical to use on more complex games, which motivates the need
              for methods that can approximate perfect play with far smaller
              computational cost.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              Turn-based games can be represented as a tree, where each node
              represents the state of the game, and each node's children consist
              of the possible states of the game after the next turn. In the
              middle of a game of tic-tac-toe, for example, the game tree might
              look something like this:
            </p>
            <ArticleImage
              src={TTTGameTree}
              caption='An example of a game tree for a tic-tac-toe board. Each node is colored according to which player makes the next
              move; on purple nodes, "X" will place the next marker, and on blue
              nodes, "O" will place the next marker. Leaf nodes, or nodes
              without any children, correspond with states in which the game has
              ended. Here, each leaf node is annotated with a value -1, 0, or
              +1, depending on which player has won the game. Our convention
              will be that a value of -1 means that "O" has won, while a value
              of +1 means that "X" has won; a value of 0 is used for a draw.
              Therefore, the purple player strives to maximize the value of the game in the final position, while
              the blue player strives to minimize it.'
              width="90%"
            />
            <p>
              Let's reason about how purple might use the values in the game
              tree to decide which move to make at the root node.
              <ul>
                <li>
                  Purple first considers the move on the left, leading to
                  position [A]. They reason that, from position [A], no matter
                  which move blue chooses, the game will result in a value of 0,
                  so they assign the value 0 to position [A].
                </li>
                <li>
                  Purple then considers the move in the center. From position
                  [B], blue has two alternatives, which lead to values of -1 and
                  +1. Since blue wants to minimize the value in the final
                  position, they will choose the move leading to a value of -1.
                  Therefore, position [B] may be assigned a value of -1.
                </li>
                <li>
                  Finally, purple considers the move on the right. From position
                  [C], blue again has two alternatives, this time leading to
                  values of -1 and 0. Blue would choose the move leading to the
                  smaller value, so position [C] may be assigned a value of -1.
                </li>
                <li>
                  Therefore, position [A] has a value of 0, position [B] has a
                  value of -1, and position [C] has a value of -1. Purple
                  reasons that among its three options, position [A] offers the
                  highest value, and elects to make the move on the left.
                </li>
              </ul>
            </p>
            <p>
              The strategy purple employed to select the optimal move was to
              consider each candidate move, and evaluate the position based on
              the opponent's best response. This strategy, known as the minimax
              algorithm, can be generalized to an arbitrary game tree.
            </p>
            <p>
              As with our tic-tac-toe example, we first define a way to
              numerically evaluate the final position in a game. It should be
              the goal of one player, known as the maximizing player, to
              maximize this evaluation, and the goal of the other player, known
              as the minimizing player, to minimize the evaluation. When
              confronting a new position, the player who would like to maximize
              the evaluation (in this case, purple) should first assign a value
              to each of its legal moves, equal to the minimum evaluation the
              opponent could hope to achieve from the new position. Among these
              moves, the maximizing player should choose the one with the
              largest value. Conversely, the minimizing player should assign a
              value to each of its legal moves, equal to the maximum evaluation
              the opponent could hope to achieve from the new position. Among
              these moves, they should choose the one with the smallest value.
            </p>
            <p>
              Before we move on to a slightly more complicated example, there's
              one more note I'd like to mention. When running minimax on complex
              games like chess and Go, the breadth and depth of the game tree
              make it infeasible to continue processing nodes until the game has
              ended. Instead, it is typical to specify a maximum search depth
              for the algorithm. Once the algorithm reaches a node at this
              depth, even if the game has not ended, the value of that node will
              be assigned based on some static evaluation of the position. For
              example, in chess, a crude heuristic that might be used to produce
              this evaluation is to take the difference between the total value
              of the white pieces on the board and the total value of the black
              pieces on the board.
            </p>
            <p>
              Let's now turn to a more complex example, represented by the
              following game tree, where we'll limit the depth of the minimax
              algorithm to 4:
            </p>
            <ArticleImage src={ArbitraryGameTree} width="80%"></ArticleImage>
            <p>
              We'll cover the first few steps of the minimax algorithm, and then
              I'll provide an animation that shows how the rest of the algorithm
              would play out.
              <ul>
                <li>
                  At the root node, purple will choose the move leading to the
                  maximum among the values of positions [A], [B], and [C]. It
                  begins by evaluating the position at node [A].
                </li>
                <li>
                  At node [A], blue will choose the move leading to the minimum
                  among the value of position [D], and the value 0. Computing
                  this minimum requires the value of node [D].
                </li>
                <li>
                  At node [D], purple will choose the move leading to the
                  maximum value among its alternatives. In this case, purple has
                  only one option, so the value of node [D] is equal to the
                  value of node [H].
                </li>
                <li>
                  At node [H], blue will choose the move leading to the minimum
                  value among +2 and +1. The value of node [H] can therefore be
                  set to +1.
                </li>
                <li>The value of node [D] can be set to +1.</li>
                <li>The value of node [A] can be set to 0.</li>
              </ul>
              The animation below shows how the algorithm would continue
              processing the nodes in the tree.
            </p>
            <ArticleImage
              src={ArbitraryGameTreeAnimation}
              alt="Animation of the minimax algorithm on a more complicated game tree"
              caption="The minimax algorithm running on a more complicated game tree, with a depth of 4. Leaf nodes are labelled with their static evaluations."
            />
            <p>
              The final evaluation of the position is 0, meaning that with
              optimal play, assuming our static evaluations are accurate, this
              game would result in a draw. The minimax algorithm can provide us
              not only with an numerical evaluation of the position, but also
              with the best continuation of the game. To determine the best
              continuation, we can start from the root node and continually
              choose the move with the maximum value if it is purple's turn, and
              the move with the minimum value if it is blue's turn.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              Minimax can be implemented in code via recursion; we'll implement
              the algorithm for our tic-tac-toe game to create an AI that can
              achieve perfect play. I would encourage you to revisit our example
              game trees after we implement the algorithm, verifying that it
              indeed finds the correct evaluation and continuation.
            </p>
            <p>
              Implementing the minimax algorithm will require adding some
              additional utilities to the <Code>Board</Code> class in{" "}
              <Code>board.py</Code>. First, we'll need a function{" "}
              <Code>get_legal_moves()</Code> so that our algorithm can find a
              node's children in the game tree. This can be implemented by
              building a list of each empty space on the board:
            </p>
            <Prism language="python">{`def get_legal_moves(self):
    legal_moves = []
    for i in range(self.board_dim):
        for j in range(self.board_dim):
            if self.board_state[i][j] == 0:
                legal_moves.append((i, j))
    return legal_moves`}</Prism>
            <p>
              We'll also need a function <Code>undo_last_move()</Code>, so that
              we can return to the original board state at a node after
              examining each candidate move. Before implementing this, we'll
              first update the <Code>reset_game()</Code> function to intialize a
              list <Code>self.move_history</Code> that keeps track of the
              history of moves:
            </p>
            <Prism language="python">
              {`def reset_game(self):
    self.board_state = np.array([[0] * self.board_dim for _ in range(self.board_dim)])
    self.move_history = []
    self.player = 1
    self.winner = None
    self.game_over = False`}
            </Prism>
            <p>
              In the <Code>make_move()</Code> function, we'll also need to add a
              line that appends the move to the move history. Update the code as
              follows:
            </p>
            <Prism language="python">
              {`def make_move(self, move):
    cell_row, cell_col = move
    self.board_state[cell_row][cell_col] = self.player
    self.update_winner()
    self.move_history.append(move)
    self.switch_player()`}
            </Prism>
            <p>
              Now, we can write the <Code>undo_last_move()</Code> function,
              where we'll pop the last move from the move history, clear the
              corresponding space on the board, and update the winner and
              current player.
            </p>
            <Prism language="python">{`def undo_last_move(self):
    cell_row, cell_col = self.move_history.pop()
    self.board_state[cell_row][cell_col] = 0
    self.update_winner()
    self.switch_player()`}</Prism>
            <p>
              Finally, we'll add a getter method that returns the move history:
            </p>
            <Prism language="python">
              {`def get_move_history(self):
    return self.move_history`}
            </Prism>
            <p>
              With all this infrastructure out of the way, we're ready to
              implement the minimax algorithm. Start by creating a new file
              called <Code>minimax.py</Code>. We'll go ahead and import the{" "}
              <Code>math</Code> and <Code>time</Code> modules, as well as our{" "}
              <Code>Board</Code> class:
            </p>
            <Prism language="python">
              {`import math
import time
from board import Board`}
            </Prism>
            <p>
              The first thing we need to do is implement a function that can
              perform a static evaluation of a position, which will enable us to
              compute the value at the leaf nodes. We'll create a function{" "}
              <Code>get_static_evaluation()</Code> that returns 1 if Player 1
              (the player with "X") wins and -1 if Player 2 (the player with
              "O") wins. If the game is a draw, or if the game has not yet
              ended, we'll return 0. We'll also want to incentive minimax to
              finish the game as quickly as possible for the winning player, and
              to prolong the game as long as possible for the losing player,
              which can be done by including an additional term in the static
              evaluation that grows with the number of moves in the game:
            </p>
            <Prism language="python">
              {`def get_static_evaluation(winner, num_moves):
    if winner == 1:
        return 1 - 0.01 * num_moves
    if winner == 2:
        return -1 + 0.01 * num_moves
    return 0
`}
            </Prism>
            <p>
              Now, if Player 1 can force a win, when following the minimax
              algorithm, Player 2 will at least try to hold on for as long as
              possible, because this will decrease the game's evaluation.
              Conversely, if Player 2 can force a win, Player 1 will try to hold
              on in order to increase the game's evaluation.
            </p>
            <p>We're now ready to define the minimax algorithm as follows:</p>
            <Prism language="python">
              {`def minimax(board, depth, maximizing_player):
    if board.get_game_over() or depth == 0:
        value = get_static_evaluation(board.get_winner(), len(board.get_move_history()))
        moves = []
    elif maximizing_player:
        value = -math.inf
        for move in board.get_legal_moves():
            board.make_move(move)
            v, m = minimax(board, depth - 1, False)
            board.undo_last_move()
            if v > value:
                value = v
                moves = [move, *m]
    else:
        value = math.inf
        for move in board.get_legal_moves():
            board.make_move(move)
            v, m = minimax(board, depth - 1, True)
            board.undo_last_move()
            if v < value:
                value = v
                moves = [move, *m]

    return value, moves`}
            </Prism>
            <p>
              The minimax algorithm accepts three parameters. The first
              parameter is a <Code>Board</Code> object <Code>board</Code>, which
              represents the state of the game. The <Code>depth</Code> parameter
              specifies the depth the algorithm should search to, starting from
              the current node. Finally, the <Code>maximizing_player</Code>{" "}
              parameter is a boolean that specifies whether or not the current
              player is the one trying to maximizing the value function. In our
              examples, it should be set to True for purple nodes and False for
              blue nodes. The algorithm should return <Code>value</Code>, a
              numerical evaluation of the current position, and{" "}
              <Code>moves</Code>, the list of moves that realizes this
              evaluation.
            </p>
            <p>
              We formulate minimax as a recursive algorithm:
              <ul>
                <li>
                  For the base case, if the game is over in the current board
                  state, or if we have reached the maximum search depth, the
                  value is simply a static evaluation of the position, which we
                  can compute using the <Code>get_static_evaluation()</Code> we
                  wrote earlier. Since no moves can be taken from the final
                  position, we can return an empty list for the list of moves.
                </li>
                <li>
                  Otherwise, if it's the turn of the maximizing player, we
                  initialize a variable <Code>value</Code> at negative infinity,
                  which will keep track of the maximium value the maximizing
                  player can guarantee. Then, we'll iterate over each legal move
                  and make the move on the board. We'll pass the new board state
                  to a recursive call to <Code>minimax()</Code>, which will
                  compute the minimum value the opponent can guarantee. Note
                  that in the recursive call, the depth decreases by one, and{" "}
                  <Code>maximizing_player</Code> is set to <Code>False</Code>,
                  because it is now the opponent's turn to move. After we undo
                  the move on the board, we check if the evaluation{" "}
                  <Code>v</Code> returned by the recursive call is better than
                  the best evaluation <Code>value</Code> we've seen so far. If
                  so, we update <Code>value</Code> to reflect the new evaluation{" "}
                  <Code>v</Code>, and build a new list <Code>moves</Code>{" "}
                  consisting of the move we just made and the move list{" "}
                  <Code>m</Code> returned by the recursive call.
                </li>
                <li>
                  Finally if it's the turn of the minimizing player, we do
                  essentially the opposite of what we did for the maximizing
                  player. We initialize <Code>value</Code> to positive infinity,
                  and for each legal move, check if the value <Code>v</Code>{" "}
                  returned by a recursive call is lower than the current best.
                  If so, we update the best value seen so far and build our list
                  of moves as before.
                </li>
              </ul>
            </p>
            <p>
              As I mentioned earlier, I would recommend revisiting our example
              boards from earlier to verify for yourself that this code does
              indeed produce the correct evaluation for each board. To run the
              code on our tic-tac-toe game, just add the following code to{" "}
              <Code>minimax.py</Code>, which will run and time the minimax
              algorithm, running from an empty tic-tac-toe board on maximum
              depth.
            </p>
            <Prism language="python">
              {`if __name__ == "__main__":
    board = Board(3)

    tic = time.perf_counter()
    value, moves = minimax(board, math.inf, True)
    toc = time.perf_counter()
    print(f"[Minimax] Time Taken: {toc - tic:0.1f} seconds | Evaluation: {value} | Best Continuation: {moves}")`}
            </Prism>
            <p>
              Running the code, you should get an output that looks something
              like this:
            </p>
            <Prism>
              {`[Minimax] Time Taken: 22.4 seconds | Evaluation: 0 | Best Continuation: [(0, 0), (1, 1), (0, 1), (0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]`}
            </Prism>
            <p>
              This output looks good! The position evaluates to 0, which means
              that tic-tac-toe is a draw under perfect play, and playing out the
              best continuation, each move seems reasonable. The only issue is
              the time taken by the algorithm, which for me, was a whopping 22.4
              seconds. Computers tend to be far better than humans at
              computationally heavy tasks, so it might be unintuitive that it
              took so long for the computer to solve the position. Slighty
              disappointed by this result, we might wonder if there is a better
              way to solve the game. It turns out that not only can we vastly
              improve on the runtime, but it also requires just one key insight
              and a few additional lines of code.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
            <p>
              To understand how we can optimize our algorithm, let's return to
              our earlier game tree example and consider the following moment in
              the minimax algorithm:
            </p>
            <ArticleImage src={AlphaBetaPruningTree} width="80%" />
            <p>
              At the root node, purple has three legal moves, and they are
              currently considering the third move. After making this move, blue
              is left with three options, the first of which has a value of +2,
              and the second of which was just evaluated to have a value of -3.
              At this point, I argue that it is a waste of time for the
              algorithm to evaluate blue's final option. To see this, consider
              that if purple selects the third move from the root, blue can
              already guarantee that the position will end up with at most a -3
              evaluation. Knowing this, purple can mark their third option as
              having an evaluation of ≤-3. However, purple's first move from the
              root yields an evaluation of 0, which is guaranteed to be better
              than their third option. Thus, there is no longer a reason to
              finish evaluating the third move.
            </p>
            <ArticleImage src={AlphaBetaPruningTreePruned} width="80%" />
            <p>
              This idea might be summarized as follows: "I can discard a
              potential move from consideration as soon I find a response from
              my opponent which proves I have a better option." To implement
              this idea in our algorithm, we need to keep track of the value of
              each's player's best option at each point in the game tree. By
              convention, we use "alpha" (α) to denote the maximum value the
              maximizing player (purple) is able to guarantee and "beta" (β) to
              denote the minimum value the minimizing player (blue) is able to
              guarantee. Thus, α should be updated whenever purple finds a new
              best move, and β should be updated whenever blue finds a new best
              move. The values α and β are passed to the recursive calls at each
              node in the tree. Now, how do we know, based on the values α and
              β, when the algorithm can safely stop considering further moves?
              Let's consider two cases: nodes on which it is purple's turn to
              move, and nodes on which it is blue's turn to move.
              <ul>
                <li>
                  Suppose the algorithm is considering a node on which it is
                  blue's turn to move. After considering a few of the potential
                  moves for blue, it eventually updates the value of β and then
                  finds that α &gt; β. The implication is that, from the current
                  node, blue can force an evaluation β, but by making a
                  different move earlier in the tree, purple can force an larger
                  evaluation α. Since purple has a better move, they will not
                  allow the game to reach the current node, and we do not need
                  to consider other options for blue.
                </li>
                <li>
                  Suppose instead that the algorithm is considering a node on
                  which it is purple's turn to move. After considering a few of
                  the potential moves for purple, it eventually updates the
                  value of α and then finds that α &gt; β. The implication is
                  that, from the current node, purple can force an evaluation α,
                  but by making a different move earlier in the tree, blue can
                  force an smaller evaluation β. Since blue has a better move,
                  they will not allow the game to reach the current node, and we
                  do not need to consider other options for purple.{" "}
                </li>
              </ul>
              Thus, in any case, if we keep the values α and β updated, whenever
              we find α &gt; β, we can break from the current call of minimax.
              This process of breaking from the current call is known as
              "pruning," because we are trimming off the computation for the
              nodes we no longer need to consider.
            </p>
            <p>
              One additional note we might make is that the order in which we
              consider moves affects how much pruning occurs. If we always
              consider the best move for each player first, it is more likely
              that the player will "prove" the opponent has a better option,
              thus meeting the pruning condition α &gt; β. To convince yourself
              of this, you can try reordering nodes in our example game tree
              such that the best options are considered first, and then stepping
              through the minimax algorithm with the alpha-beta pruning
              optimization. Therefore, when implementing alpha-beta pruning on
              games like chess, considering moves that are likely to be good,
              such as checks, captures, and attacks, earlier on, can help speed
              up the algorithm by increasing the amount of pruning that can
              occur.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[4]} />
            <p>
              Implementing the additional code for alpha-beta pruning on top of
              our base minimax algorithm should be fairly simple, so I'd
              recommend trying it for yourself before looking at the solution.
            </p>
            <p>
              To add in alpha-beta pruning, let's first copy over our old{" "}
              <Code>minimax()</Code> function, which is worth keeping around so
              that we can compare the performance of the two algorithms. We'll
              rename the function to <Code>pruned_minimax</Code> and add
              parameters <Code>alpha</Code> and <Code>beta</Code> to the
              function signature so that their current values can be passed to
              the recursive calls. Then, in the <Code>maximizing_player</Code>{" "}
              case, after considering each move, we'll update the value of{" "}
              <Code>alpha</Code> if necessary, and we'll then break from the
              call if <Code>{`beta <= alpha`}</Code>. In the{" "}
              <Code>minimizing_player</Code> case, after considering each move,
              we'll update the value of <Code>beta</Code> if necessary, again
              breaking from the call if <Code>{`beta <= alpha`}</Code>.
            </p>
            <Prism language="python">
              {`def pruned_minimax(board, depth, alpha, beta, maximizing_player):
    moves = []
    if board.get_game_over() or depth == 0:
        value = get_static_evaluation(board.get_winner(), len(board.get_move_history()))
    elif maximizing_player:
        value = -math.inf
        for move in board.get_legal_moves():
            board.make_move(move)
            v, m = pruned_minimax(board, depth - 1, alpha, beta, False)
            board.undo_last_move()
            if v > value:
                value = v
                moves = [move, *m]
            alpha = max(alpha, value)
            if beta <= alpha:
                break
    else:
        value = math.inf
        for move in board.get_legal_moves():
            board.make_move(move)
            v, m = pruned_minimax(board, depth - 1, alpha, beta, True)
            board.undo_last_move()
            if v < value:
                value = v
                moves = [move, *m]
            beta = min(beta, value)
            if beta <= alpha:
                break

    return value, moves`}
            </Prism>
            <p>
              We'll update the code that times the minimax algorithm to include
              a test of our alpha-beta pruning version:
            </p>
            <Prism language="python">{`if __name__ == "__main__":
    board = Board(3)

    tic = time.perf_counter()
    value, moves = minimax(board, math.inf, True)
    toc = time.perf_counter()
    print(f"[Minimax] Time Taken: {toc - tic:0.1f} seconds | Evaluation: {value} | Best Continuation: {moves}")

    tic = time.perf_counter()
    value, moves = pruned_minimax(board, math.inf, -math.inf, math.inf, True)
    toc = time.perf_counter()
    print(f"[Pruned Minimax] Time Taken: {toc - tic:0.1f} seconds | Evaluation: {value} | Best Continuation: {moves}")`}</Prism>
            <p>
              Running the code, the time saved by the optimization is clear; my
              results show the following:
            </p>
            <Prism>{`[Minimax] Time Taken: 22.7 seconds | Evaluation: 0 | Best Continuation: [(0, 0), (1, 1), (0, 1), (0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]
[Pruned Minimax] Time Taken: 0.9 seconds | Evaluation: 0 | Best Continuation: [(0, 0), (1, 1), (0, 1), (0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]`}</Prism>
            <ArticleHeader sectionHeader={sectionHeaders[5]} />
            <p>
              To understand why we might need more sophisticated methods of
              analyzing positions, go ahead and instantiate a board with
              dimension 4--<Code>board = Board(4)</Code>--and try running the
              minimax algorithm with alpha-beta pruning. After running the code
              on my own machine for about half an hour, and the algorithm still
              hadn't finished. Just by increasing the width and height of the
              board by one dimension, we can observe our algorithm struggling to
              solve the position.
            </p>
            <p>
              It turns out that for sufficiently complex games, it isn't always
              feasible to use an algorithm that tries to play the game
              perfectly. Instead, we'll need to develop algorithms that can
              approximate perfect play, such as Monte Carlo tree search, which
              will be the subject of the next article in this series.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[6]} />
            <p>
              It's cool to see our algorithm working (and doing so fairly
              quickly), but it would be awesome to if our algorithm could tell
              us the best continuation from a position in the middle of a game.
              Our final order of business will be to add the board evaluation
              computed by our algorithm to the display we've implemented.
            </p>
            <p>
              To begin, we'll add one more getter method to the{" "}
              <Code>Board</Code> class. We need to be able to retrieve the
              current player in order to set <Code>maximizing_player</Code> for
              the initial call to minimax:
            </p>
            <Prism language="python">
              {`def get_player(self):
    return self.player`}
            </Prism>
            <p>
              Each time a move is made on the tic-tac-toe board, we'll need to
              run minimax to get the new evaluation. These calls to minimax will
              be spawned from the <Code>BoardDisplay</Code> class. To top of{" "}
              <Code>board_display.py</Code>, import the <Code>math</Code> module
              (so we have access to <Code>math.inf</Code>), the
              <Code>copy</Code> module (so that we can copy the board state),
              and the <Code>Thread</Code> class from the <Code>threading</Code>{" "}
              module (so that we can make a call to the minimax algorithm
              without freezing the board while it's running). Finally, import
              our own <Code>pruned_minimax</Code> function. Your imports should
              now look like this:
            </p>
            <Prism language="python">
              {`import pygame
from pygame.locals import *
import math
import copy
from threading import Thread

from board import Board
from minimax import pruned_minimax`}
            </Prism>
            <p>
              To the <Code>BoardDisplay</Code> class, we'll add functions{" "}
              <Code>update_evaluation()</Code>, <Code>run_evaluation()</Code>,
              and <Code>get_evaluation()</Code>.
            </p>
            <Prism language="python">
              {`def update_evaluation(self):
    self.evaluation = None
    self.best_continuation = None
    t = Thread(target=self.run_evaluation)
    t.start()


def run_evaluation(self):
    self.evaluation, self.best_continuation = pruned_minimax(copy.deepcopy(self.board), math.inf, -math.inf, math.inf, self.board.get_player() == 1)


def get_evaluation(self):
    return self.evaluation, self.best_continuation`}
            </Prism>
            <p>
              The <Code>update_evaluation()</Code> function is responsible for
              resetting the current estimates of the evaluation and best
              continuation, and then starting a thread to run the{" "}
              <Code>run_evaluation()</Code> function, which will invoke minimax
              and upate these values. Finally, <Code>get_evaluation()</Code> is
              a combined getter method for the evaluation and the best
              continuation.
            </p>
            <p>
              We'll need one more method to return a copy of the board, which
              will be used later to render the boards used in displaying the
              best continuation of the current position.
            </p>
            <Prism language="python">
              {`def get_board(self):
    return copy.deepcopy(self.board)`}
            </Prism>
            <p>
              At the end of the constructor for <Code>BoardDisplay</Code>, we'll
              want to make our first evaluation of the main board. Add the
              following code:
            </p>
            <Prism language="python">
              {`if self.is_main_board:
    self.update_evaluation()`}
            </Prism>
            <p>
              We'll also want to update <Code>draw_board()</Code> to update the
              evaluation in two places: when the board is reset, and when a move
              is made. With these additional calls to{" "}
              <Code>update_evaluation()</Code>, the draw board function should
              look like this:
            </p>
            <Prism language="python">
              {`def draw_board(self, clicked=False, pos=None):
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
        self.draw_winner()`}
            </Prism>
            <p>
              Now, to display the evaluation, we'll create a new class{" "}
              <Code>EvaluationDisplay</Code>. Like the <Code>BoardDisplay</Code>{" "}
              class, this class will accept a surface as a parameter in the
              constructor, onto which it will draw. Then, the{" "}
              <Code>MainDisplay</Code> class will be responsible for rendering
              this surface onto the main screen. Add a file called{" "}
              <Code>evaluation_display.py</Code> with the following code:
            </p>
            <Prism language="python">
              {`import pygame
from pygame.locals import *

from board_display import BoardDisplay


class EvaluationDisplay:
    def __init__(self, surface, board_dim, evaluation_size):
        self.surface = surface
        self.board_dim = board_dim
        self.evaluation_size = evaluation_size

        self.font = pygame.font.SysFont("Courier", self.evaluation_size // 15)

        self.bg_color = (15, 15, 15)
        self.text_color = (255, 255, 255)


    def draw_evaluation(self, board, evaluation, best_continuation):
        self.surface.fill(self.bg_color)

        if evaluation is None:
            computing_text = "Computing..."
            computing_img = self.font.render(computing_text, True, self.text_color)
            self.surface.blit(computing_img, (0, 0))
            return

        # Draw evaluation text
        evaluation_text = f"Evaluation: {evaluation:.2f}"
        evaluation_img = self.font.render(evaluation_text, True, self.text_color)
        self.surface.blit(evaluation_img, (0, 0))

        # Draw continuation text
        continuation_text = f"Best Continuation: {'...' if best_continuation is None else ''}"
        continuation_img = self.font.render(continuation_text, True, self.text_color)
        self.surface.blit(continuation_img, (0, self.evaluation_size // 8))

        # Draw continuation boards
        if best_continuation is not None:
            for i, move in enumerate(best_continuation[:9]):
                board.make_move(move)
                board_size = self.evaluation_size // 5
                board_surface = pygame.Surface((board_size, board_size))
                board_display = BoardDisplay(board_surface, self.board_dim, board_size, False, board)
                board_display.draw_board()
                x_pos = (i % 3 * self.evaluation_size // 4)
                y_pos = self.evaluation_size // 4 + (i // 3 * self.evaluation_size // 4)
                self.surface.blit(board_surface, (x_pos, y_pos))`}
            </Prism>
            <p>
              The <Code>draw_evaluation</Code> function accepts parameters for
              the current game board, the current evaluation, and the best
              continuation of the position. It renders the evaluation score to
              the drawing surface, and it displays the best continuation of the
              game by making each move in <Code>best_continuation</Code> and
              initializing <Code>BoardDisplay</Code> objects with the current
              board and with <Code>is_main_board</Code> set to{" "}
              <Code>False</Code>.
            </p>
            <p>
              Finally, in the <Code>MainDisplay</Code> class, in each pass of
              the game loop, we'll want to get the latest evaluation from the{" "}
              <Code>BoardDisplay</Code>, pass it to{" "}
              <Code>draw_evaluation()</Code>, and render the{" "}
              <Code>EvaluationDisplay</Code>. We'll first need to import our{" "}
              <Code>EvaluationDisplay</Code> class:
            </p>
            <Prism language="python">
              {`from evaluation_display import EvaluationDisplay`}
            </Prism>
            <p>
              Then, in the constructor, we'll change the call to{" "}
              <Code>pygame.display.set_mode()</Code> to use a 2:1 ratio, which
              will give us room for the evaluation display. We'll also adjust
              the position and size of the <Code>BoardDisplay</Code>, and we'll
              initialize our <Code>EvaluationDisplay</Code>:
            </p>
            <Prism language="python">
              {`def __init__(self, board_dim, window_width):
    pygame.init()

    # Set up main window
    self.window_width = window_width

    self.bg_color = (15, 15, 15)

    self.screen = pygame.display.set_mode((self.window_width, self.window_width // 2))
    self.screen.fill(self.bg_color)

    pygame.display.set_caption('Tic-Tac-Toe')

    # Set up main board display
    self.main_board_size = 5 * self.window_width // 12
    self.main_board_pos = (self.window_width // 18, self.window_width // 24)
    self.main_board_surface = pygame.Surface((self.main_board_size, self.main_board_size))
    self.main_board_display = BoardDisplay(self.main_board_surface, board_dim, self.main_board_size, True)

    # Set up evaluation display
    self.evaluation_size = 5 * self.window_width // 12
    self.evaluation_pos = (19 * self.window_width // 36, self.window_width // 24)
    self.evaluation_surface = pygame.Surface((self.evaluation_size, self.evaluation_size))
    self.evaluation_display = EvaluationDisplay(self.evaluation_surface, board_dim, self.evaluation_size)
`}
            </Prism>
            <p>
              Finally, in <Code>run_game()</Code>, we'll retrieve and draw the
              latest evaluation.
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

        # Retrieve and draw latest evaluation
        board = self.main_board_display.get_board()
        evaluation, best_continuation = self.main_board_display.get_evaluation()
        self.evaluation_display.draw_evaluation(board, evaluation, best_continuation)
        self.screen.blit(self.evaluation_surface, self.evaluation_pos)

        pygame.display.update()

    pygame.quit()`}
            </Prism>
            <p>
              If all has gone well, you should now be able to view the
              evaluation and best continuation of the tic-tac-toe board as you
              play!
            </p>
            <img
              src="https://media.giphy.com/media/g6i3d7p3Jtg9wYqgnp/giphy.gif"
              alt="GIF of Playing Tic-Tac-Toe on a Board with Live Evaluation"
              width="75%"
            />
          </div>
        </div>
      </div>
    </>
  );
};

export default MinimaxPage;

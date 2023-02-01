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
              {/* MCTS converges to an optimal solution by improving its predictions
              over time. As nodes are added to the game tree in the expansion
              phase, more of the game is played out in the selection phase and
              less of the game is played out in the simulation phase, where
              moves are selected uniformly at random. In the selection phase, we
              can select moves according to a policy, so that optimal lines of
              play will be explored more often than suboptimal ones. */}
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default MCTSPage;

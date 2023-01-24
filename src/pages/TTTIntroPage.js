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
              chess, it isn't unreasonable to be surprised that it wasn't until
              two decades after Kasparov's defeat that a computer was able to
              defeat a 9-dan professional Go player. DeepMind's AlphaGo won a
              five-game match against Lee Sedol in 2016, marking another
              milestone in the progress of aritifical intelligence. At this
              point, I owe you answers to at least a couple of questions:
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
                of defeating Lee Sedol?
              </li>
            </ol>
            <p>
              Of course, these questions are quite complex, so if we want to
              answer them, we'll need to start with something much simpler. Like
              chess and Go, tic-tac-toe is a game of perfect information, and it
              will be our game of choice for learning about how we make AI for
              games. We'll be working with tic-tac-toe for several reasons:
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
                  <strong>Scalable: </strong>We can easily scale the tic-tac-toe
                  board to dimensions larger than 3x3 to give our AI a more
                  complex task.
                </li>
                <li>
                  <strong>Non-trivial: </strong>Though tic-tac-toe can be fully
                  "solved" by a human, it certainly isn't trivial to write a
                  program capable of solving the game in a reasonable amount of
                  time. In other words, it will still be gratifying to see that
                  our code works!
                </li>
                <li>
                  <strong>Similarities with Go: </strong>While tic-tac-toe is
                  far less complex than Go, the two games share many
                  elements--they are both turn-based games of perfect
                  information that are played on a grid--so it should be more or
                  less obvious how our algorithms implemented for tic-tac-toe
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
            {/* Describe how I originally implemented it. */}
          </div>
        </div>
      </div>
    </>
  );
};

export default TTTIntroPage;

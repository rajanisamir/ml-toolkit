import React from "react";
import { useRef } from "react";

import Eq from "../components/Eq";
import BlockEq from "../components/BlockEq";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleImage from "../components/ArticleImage";

import EncoderDiagram from "../images/EncoderDiagram.svg";
import MultiHeadAttentionDiagram from "../images/MultiHeadAttentionDiagram.svg";
// import AttentionDiagram from "../images/AttentionDiagram.svg";

const TransformerPage3 = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Training a Transformer Model",
      id: "training",
    },
    {
      name: "Masked Multi-Head Attention",
      id: "masked-multi-head-attention",
    },
    {
      name: "The Decoder Architecture",
      id: "decoder-architecture",
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
            <ArticleTitle name={"The Transformer"} />
            <ArticleSubtitle name={"Part 3: The Decoder"} />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              To motivate the architecture of the Transformer's decoder, we
              first need to think carefully about how we will be training it,
              which is something we've neglected discussing so far. First, let's
              recall how a neural network is typically trained on a
              classification task, such as digit recognition on the MNIST
              dataset. Training samples in the MNIST dataset are handwritten
              digits labelled 0-9. If we want to train our neural network with
              gradient descent, we need a differentiable loss function to
              determine in which direction the network's parameters should be
              adjusted. If the network simply outputted a single number
              indicating the predicted label, the only information we'd glean is
              whether or not the prediction was correct, and it would not be
              possible to come up with a suitable loss function. Instead, we
              typically require the output layer of the network to be a
              probability distribution over the possible labels. We map the
              ground truth to a one-hot vector (e.g. a label of 3 would map to
              the vector <Eq text="$[0,0,0,1,0,0,0,0,0,0]$" />
              ), and we take the mean squared error between our output vector
              and the target vector to be the loss. Now, smoothly decreasing
              this loss simply involves increasing the probability of the target
              digit in the network's output vector and decreasing the
              probability of each other digit in the network's output vector.
              Similarly, for a task where the neural network outputs a word from
              a corpus (i.e. the set of all possible words in a lanuage), we
              don't just want the neural network to output a single word, but
              rather a probability distribution over the possible words. For
              example, if a language has 50,000 words, we want the output layer
              to be a 50,000-dimensional vector, where each dimension is the
              probability of a particular word.
            </p>
            <p>
              Now, what happens when we want the model to output a full
              sentence? As before, let's suppose we're using the Transformer for
              a English-to-Spanish language translation task.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default TransformerPage3;

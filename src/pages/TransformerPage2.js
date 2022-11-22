import React from "react";
import { useRef } from "react";

import { Anchor } from "@mantine/core";

import Eq from "../components/Eq";
// import BlockEq from "../components/BlockEq";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleFeedback from "../components/ArticleFeedback";
import ArticleImage from "../components/ArticleImage";

import EncoderDiagram from "../images/EncoderDiagram.svg";
// import AttentionDiagram from "../images/AttentionDiagram.svg";

const TransformerPage2 = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Preface: Feedback",
      id: "preface",
    },
    {
      name: "Recap & Where We're Headed",
      id: "recap",
    },
    {
      name: "Multi-Head Attention",
      id: "multi-head-attention",
    },
    {
      name: "Feedforward Network",
      id: "transformer-embeddings-and-attention",
    },
    {
      name: "Residual Connections",
      id: "residual-connections",
    },
    {
      name: "Repeating the Encoder Block",
      id: "repeat",
    },
    {
      name: "Positional Encoding",
      id: "transformer-dot-product-attention",
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
            <ArticleSubtitle name={"Part 2: Multi-Head Attention & the Encoder"}/> 
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              Because this article will be the first one published on this
              website, I'd like to gather some feedback from the readers. There
              is a phenomenon known as the{" "}
              <Anchor
                href="https://en.wikipedia.org/wiki/Curse_of_knowledge"
                target="_blank"
              >
                curse of knowledge
              </Anchor>
              , which says, roughly, that those who already know about a topic
              have a hard time understanding the mindset of those who don't, and
              thus often struggle to effectively teach it. My goal here is to
              avoid falling into this trap as much as possible, and the only way
              to guarantee this is by soliciting feedback from the readers.
              Here, you can let me know what you struggled through, or which
              explanations you found confusing, and I'll do my best to adjust
              the content as needed.
            </p>
            <ArticleFeedback />
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              Let's zoom out from our discussion in the previous section and take a look at the big picture of Transformer's encoder module, just to give you an idea of where we're headed.
            </p>
            <ArticleImage src={EncoderDiagram} width="50%" caption="The Transformer's encoder might look scary right now, but by the end of this section, you'll understand the function of each of its components!"/>
            <p>
              In the previous section, we covered the attention mechanism, which involved mapping each word embedding to a query, key, and value, and then judiciously combining those vectors to produce a context-aware embedding. This mechanism is conceptually the toughest part of the Transformer to understand, and we'll now turn our focus to adding in several of the simpler bits and pieces that make up the Transformer's encoder. First, we'll expand on scaled dot-product attention to develop a slightly more sophisticated mechanism known as multi-head attention. We'll then cover the inclusion of a feedforward network, residual connections, and the repetition of the encoder block, all of which build on the network's architecture in fairly straightforward ways. Finally, we'll finish off the encoder by discussing why it is necessary to include a positional encoding in the word embeddings and how this is accomplished in the Transformer.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
            To motivate multi-head attention, let's go back to our example sentence, "I made up a story." When translating our sentence into its intermediate representation, you might imagine that two different words need to pay attention to the word "a." First, the intermediate representation of "made" should somehow encode the information that this word is refering to fabrication, rather than something like creation (e.g. "I made up with him."), and thus paying attention to "a" is probably important. Second, the word "story" might need to pay attention to "a" so that its intermediate representation encodes it not as a particular story that has been referred to before (i.e. "the story"), but as a generic story. In this example, the word "a" is serving two completely different purposes, but the same exact value vector will get added (after being scaled by some attention score) to the embeddings of both "made" and "story." What we've realized here is that even though one word might need to pay attention to another word for a completely different reason, our current scaled dot-product attention model would have the same effect on both embeddings. Really, a given word should have multiple attention scores for each word in an input sentence, each specifying to what extent the word should pay attention to each other word, for each of several different reasons. Multi-head attention accomplishes solves exactly this issue!
            </p>
            <p>
            Recall that in scaled dot-product attention, we calculated the context-aware embedding by taking a dot product between a single 512-dimensional query and a single 512-dimensional key, multiplying the resulting scalar attention score with a single 512-dimensional value. Now, however, we want to compute multiple attention scores for each query-key pair. We can accomplish this by mapping each query, key, and value to several other vectors, each via a different linear transformation. To keep the computational cost roughly the same, each of these vectors should be of a smaller dimension. For concreteness, we'll project each query, key, and value to eight different 64-dimensional vectors. Thus, using a total of 24 different linear transformations, we can now compute eight different attention scores, each of which corresponds with eight different values, which represent eight different effects each of the words in a sentence might have on the query word. Each of the attention layers, which involves three different linear projections (one each for the query, key, and value) is referred to as an "attention head," and here we have taken the number of heads to be <Eq text="$$h=8$$"/>, the dimension of the queries, keys, and values before projection to be <Eq text="$$d_{\text{model}}=512$$"/>, and their dimensions after projection to be <Eq text="$$d_{\text{model}}/h=64$$"/>.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default TransformerPage2;

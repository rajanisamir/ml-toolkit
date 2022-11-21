import React from "react";
import { useRef } from "react";

import { Anchor } from "@mantine/core";

// import Eq from "../components/Eq";
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
              In the previous section, we covered the attention mechanism, which involved mapping each word embedding to a query, key, and value, and then judiciously combining those vectors to produce a context-aware embedding. This mechanism is conceptually the toughest part of the Transformer to understand, and this section will focus on adding in several bits and pieces that make up the Transformer's encoder. First, we'll expand on scaled dot-product attention to develop a slightly more sophisticated mechanism known as multi-head attention. We'll then cover the inclusion of a feedforward network, residual connections, and the repetition of the encoder block, all of which build on the network's architecture in fairly straightforward ways. Finally, we'll finish off the encoder by discussing why it is necessary to include a positional encoding in the word embeddings and how this is accomplished in the Transformer.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
          </div>
        </div>
      </div>
    </>
  );
};

export default TransformerPage2;

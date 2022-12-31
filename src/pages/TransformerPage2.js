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
import MultiHeadAttentionDiagram from "../images/MultiHeadAttentionDiagram.svg"
// import AttentionDiagram from "../images/AttentionDiagram.svg";

const TransformerPage2 = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Recap & Where We're Headed",
      id: "recap",
    },
    {
      name: "Multi-Head Attention",
      id: "multi-head-attention",
    },
    {
      name: "Feed Forward Network",
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
            <ArticleImage src={EncoderDiagram} width="40%" caption="The Transformer's encoder might look scary right now, but by the end of this section, you'll understand the function of each of its components!" float="right" />   
            <p>
              Let's zoom out from our discussion in the previous section and take a look at the big picture of Transformer's encoder module, just to give you an idea of where we're headed. In the previous section, we covered the attention mechanism, which involved mapping each word embedding to a query, key, and value, and then judiciously combining those vectors to produce a context-aware embedding. This mechanism is conceptually the toughest part of the Transformer to understand, and we'll now turn our focus to adding in several of the simpler bits and pieces that make up the Transformer's encoder. First, we'll expand on scaled dot-product attention to develop a slightly more sophisticated mechanism known as multi-head attention. We'll then cover the inclusion of a feed forward network, residual connections, and the repetition of the encoder block, all of which build on the network's architecture in fairly straightforward ways. Finally, we'll finish off the encoder by discussing why it is necessary to include a positional encoding in the word embeddings and how this is accomplished in the Transformer.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
            To motivate multi-head attention, let's go back to our example sentence, "I made up a story." When translating our sentence into its intermediate representation, you might imagine that two different words need to pay attention to the word "a." First, the intermediate representation of "made" should somehow encode the information that this word is refering to fabrication, rather than something like reconciliation (e.g. "I made up with him."), and thus paying attention to "a" is probably important. Second, the word "story" might need to pay attention to "a" so that its intermediate representation encodes it not as a particular story that has been referred to before (i.e. "the story"), but as a generic story. In this example, the word "a" is serving two completely different purposes, but the same exact value vector will get added (after being scaled by some attention score) to the embeddings of both "made" and "story." What we've realized here is that even though one word might need to pay attention to another word for a completely different reason, our current scaled dot-product attention model would have the same effect on both embeddings. Really, a given word should have multiple attention scores for each word in an input sentence, each specifying to what extent the word should pay attention to each other word, for each of several different reasons. Multi-head attention accomplishes solves exactly this issue.
            </p>
            <p>
            Recall that in scaled dot-product attention, we calculated the context-aware embedding by taking a dot product between a single 512-dimensional query and a single 512-dimensional key, multiplying the resulting scalar attention score with a single 512-dimensional value. Now, however, we want to compute multiple attention scores for each query-key pair, and each of these attention scores should have a different associated value to represent its effect on translation. To accomplish this, we'll map each given query to several smaller-dimensional queries, each given key to several smaller-dimensional keys, and each given value to several smaller-dimensional values. For concreteness, we'll project each 512-dimensional query, key, and value to eight different 64-dimensional queries, keys, and values. Thus, even though we've increased the number of queries, keys, and values for each word, the smaller dimension of each one means the computational cost should be roughly the same. Formally, each set of query, key, and value projections is referred to as an "attention head," and each attention head allows us to compute a distinct attention score for a given ordered pair of words. Here we have taken the number of heads to be <Eq text="$$h=8$$"/>, the dimension of the queries, keys, and values before projection to be <Eq text="$$d_{\text{model}}=512$$"/>, and their dimensions after projection to be <Eq text="$$d_q = d_k = d_v = d_{\text{model}}/h=64$$"/>.
            </p>
            <ArticleImage src={MultiHeadAttentionDiagram} width="40%"/>  
            <p>
            After each of the eight smaller-dimensional queries, keys, and values are fed through the scaled dot-product attention block, we end up with eight different context-aware embeddings. The Transformer architecture combines these embeddings by concatenating them into a single vector, and then feeding that vector through a linear layer for further processing. This process of splitting the queries, keys, and values into multiple vectors before recombining the context-aware embeddings into a single vector is referred to as "multi-head attention."
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              After mapping a word embedding to a context-aware embedding, the Transformer includes a feed forward layer, which allows for further processing of the vectors. The original paper used two consecutive linear transformations: the context-aware embedding with <Eq text="$$d_{\text{model}}=512$$"/> was expanded to dimension <Eq text="$$d_{\text{ff}}=2048$$"/> and then shrunk again to dimension <Eq text="$$d_{\text{model}}=512$$"/>. 
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
            <p>
              Residual connectitons, or skip connections, were popularized by the paper "Deep Residual Learning for Image Recognition" (He et al. 2015), which achieved state-of-the-art results on computer vision tasks using a ResNet, or a deep neural network augmented by residual connections. To understand why the Transformer incorporates these residual connections to improve its performance, we'll look at an experiment from the He et al. paper. The authors trained both a 20-layer and a 56-layer deep neural network on an image classification task using the CIFAR-10 dataset. They observed that, during training, the error of the 56-layer network was consistently higher than that of the 20-layer network, which seems counterintuitive: since the deeper network emcompasses the shallower one, but simply with additional parameters available for tuning, shouldn't its performance be at least as good? Indeed, if the additional layers in the deeper network simply consisted of an identity function, the performance would be equal to that of the shallower network.
            </p>
            <p>
              Inspired by the results of these experiments, residual connections can be thought of as a means of explicitly constructing the identity function between layers, thereby alleviating the issue. In particular, a residual connection takes a vector <Eq text="$$x$$"/> from some part of the network and adds it to some future output of the network <Eq text="$$\mathcal{F}(x)$$"/>, one or more layers later. Since the identity mapping is added to whatever function of the vector <Eq text="$$x$$"/> is computed by the neural network, the identity mapping can be thought of as "built-in" to the network. With this modification, deeper neural networks have an easier time training: the inclusion of skip connections means they should be at least as good as their shallower counterparts. The Transformer includes residual connections, followed by normalization, from before to after the multi-head attention block, and from before to after the feed forward network, to improve its performance.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[4]} />
            <p>
              After one pass through the encoder, each word in the input sentence has been turned into an intermediate representation, formed by paying attention to other input words. You might imagine, however, that certain contextual information can only be learned when an input word pays attention to a representation that has <em>already</em> been imbued with context. In other words, multiple passes through the attention mechanism might facilitate the learning of deeper contexual information. The Transformer realizes this intuition by repeating the encoder block several times; after each pass through the encoder, intermediate representations pick up more context. The original paper found that repeating the encoder block <Eq text="$$N=6$$"/> times worked well, but this is a hyperparameter that may be tuned.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[5]} />
            <p>
              Consider the input sentences "Alice likes her dog" and "Her dog likes Alice." Since these two sentences have different meanings, we would also expect that the context dependencies among words in the first input sentence would differ from those in the second. However, with our current architecture, each word in the first sentence would be processed in the same way as each word in the second sentence. Since the sentences contain the same exact words, and because the only factor that affects how a word is processed, aside from parameters of the neural network, are which other words are in the input sentence, both sentences would produce precisely the same context-aware embeddings. How do we incoporate information about the relative positions of words into the Transformer?
            </p>
            <p>
              "Attention is All You Need" found that this issue could be handled by something known as a "positional encoding." Recall that the first step in our architecture involved mapping each word to a 512-dimensional vector known as a "word embedding." The idea of the positional encoding is to imbue each word embedding with additional context related to its position within a sentence by adding a special vector to it. In principle, this vector could computed using any positional encoding function <Eq text="$$PE$$"/> that takes as input the position <Eq text="$$pos$$"/> of the word within the input sentence and the embedding dimension (i.e. the position along a vector) <Eq text="$$i$$"/> and outputs a real number, which is the number that would be added <Eq text="$$i$$"/>th dimension of the word embedding for the word in position <Eq text="$$pos$$"/> within the input sentence. The specific positional encoding function used by the original paper is alternates for odd and even embedding dimensions as follows:              
            </p>
            <BlockEq
              text="$$
                \begin{aligned}
                  PE_{(pos, 2i)} &= sin(pos/10000^{2i/d_{model}})\\
                  PE_{(pos, 2i + 1)} &= cos(pos/10000^{2i/d_{model}})
                \end{aligned}
              $$"
              displayMode={true}
            />
            <p>
              The details of how this function was designed are beyond the scope of this post, and the specific function is not covered in much detail in the original paper. In any case, the authors found that adding some computed vector to each word embedding was sufficient to provide the Transformer with positional information.
            </p>
            <p>
              With this, we're finished with the details of the encoder, and it's worth reviewing the architecture again. The encoder begins by mapping each word to a word embedding, or a vector representation of that word. Then, a positional encoding function computes a vector for each word in the input, which is added to the word embedding to produce a position-aware embedding. The position-aware embedding is fed through a multi-head attention block, the output of which is added with the input via a residual connection, and the resulting vector is normalized. Finally, this vector, which may be thought of as a context-aware embedding, is fed through a feed forward network. The output of the feed forward network undergoes an addition with its input through a residual connection, and the sum is normalized. Finally, the outputs are fed back through the encoder block several more times to produce the final intermediate representations.
            </p>
            <ArticleImage src={EncoderDiagram} width="50%"/>   
          </div>
        </div>
      </div>
    </>
  );
};

export default TransformerPage2;

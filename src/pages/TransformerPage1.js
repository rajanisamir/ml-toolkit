import React from "react";
import { useRef } from "react";

import { Anchor, Blockquote } from "@mantine/core";

import Eq from "../components/Eq";
import BlockEq from "../components/BlockEq";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleFeedback from "../components/ArticleFeedback";
import ArticleImage from "../components/ArticleImage";

import QKVDiagram from "../images/QKVDiagram.svg";
import AttentionDiagram from "../images/AttentionDiagram.svg";

import Translation from "../images/Translation.svg"

const TransformerPage1 = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Preface: Feedback",
      id: "preface",
    },
    {
      name: "Background & Motivation",
      id: "transformer-background",
    },
    {
      name: "Word Embeddings & the Attention Mechanism",
      id: "transformer-embeddings-and-attention",
    },
    {
      name: "Scaled Dot-Product Attention",
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
            <ArticleSubtitle name={"Part 1: Paying Attention"}/> 
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
              The Transformer is a neural network architecture that was first
              introduced in 2017 by Vaswani et al. in the paper
              "Attention is All You Need." Since its introduction, the
              model has spurred on major advancements in domains such as natural
              langauge processing and computer vision; thus, understanding not
              only its use cases, but also how it works under the hood, is
              invaluable for machine learning engineers and hobbyists alike. The
              Transformer is a model designed for sequence transduction, meaning
              both the input and the output of the model are some sequence of
              tokens. As an example, suppose we wanted to design a model that
              could take as input an English sentence (say, the sentence, "I
              made up a story") and output its Spanish translation ("Inventé una
              historia"). Despite how naturally language comes to humans, it's
              no secret that such a task is non-trivial for a machine; if you've
              taken a course in a foreign language, you've likely joked about
              how useless Google Translate could often be (though nowadays it's quite good). In this post, I'll
              try to demystify the Transformer by trying to convince you that,
              as a machine learning researcher with enough time and patience,
              you might have come up with a similar idea yourself. Let's begin
              by dissecting our toy example to better understand why this task
              can quickly become quite complicated.
            </p>
            <ArticleImage src={Translation} width="85%"/>
            <p>
              A naive language translation program might attempt to translate
              our sentence, "I made up a story," word-by-word. Our program might
              begin by translating the word "I" to "yo."

                  If you've taken an
                  introductory Spanish course, you probably know the inclusion of
                  the pronoun "yo" can sometimes be  superfluous, because the conjugation
                  of the verb that follows would reveal that the sentence is in
                  first-person. Certainly, however, what our program has done isn't
                  incorrect. 

               So far, so good. The next word, "made," the conjugated
              form of the verb "to make," poses issues, though. The first issue
              you might notice is that the verb "to make" in English is
              typically translated to "hacer" is Spanish. Our example, however,
              uses the phrase "made up," a phrase in which it <em>wouldn't</em>{" "}
              make sense to use "hacer." The English verb "make" can have more
              than one Spanish translation depending on the context. The verbs
              used in the Spanish translations of "I made up with someone," "I
              made a cake," and "I made up a story" would all be different. In
              the first sentence, the meaning captured by "made" is related to
              reconciliation, in the second sentence, creation, and in the
              third, fabrication. If only the program had known the implications
              of the word "up" on the translation of the word "made", it could
              deal with our translation task, because it could understand that,
              in context, we're talking about fabrication. Another issue is that
              the program doesn't know what conjugation "made" should translate
              to in Spanish. The word "made" serves as the past-tense
              conjugation for every pronoun in English: the first-person
              singular ("I made") and plural ("we made"), second-person singular
              ("you made") and plural ("y'all made"), and third-person singular
              ("they/he/she made") and plural ("they made"). In Spanish,
              however, the verb's conjugated form depends on the preceding
              pronoun: yo hice, tú hiciste, él/ella/usted hizo, nosotros
              hicimos, vosotros hicisteis, ellos/ellas/ustedes hicieron.
              Fundamentally, we have the same issue we had when choosing the
              verb itself: without context, there does not exist a one-to-one
              mapping from English to Spanish of the word "made". The key here
              is <em>context</em>: if only there were some way for a program to
              understand which words in a sentence should affect the
              translations of others, it might be able to deal with this task.
              We might separate this task of context into two sub-tasks: first,
              for each word in a sentence, our model must understand{" "}
              <em>which</em> other words affect how it should be translated;
              second, for each word, the model must understand <em>how</em>{" "}
              those other words affect how it should be translated. If these two
              sub-tasks are implemented correctly, words should be able to
              fruitfully <em>pay attention</em> to other words during a
              translation task.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              Before we get into the details of developing a model that
              facilitates such a mechanism of <em>attention</em>, we ought to
              take a step back and decide how the words in an input sentence
              should be represented in the computer. If you have experience with
              neural networks, you probably know that the canonical way to
              represent data is using a vector. An image is nothing but a vector
              of pixel values; a waveform audio file is just a vector of
              amplitudes at different points in time. It would seem appropriate
              to represent words as vectors, too, but it's much less clear how
              we would do so: what word should correspond with what vector?
              Well, the short answer is, it doesn't matter. The slightly longer
              answer is that, when we train the neural network, it can{" "}
              <em>learn</em> how to best represent each word. It certainly
              shouldn't be readily apparent to you how our network will learn
              these embeddings, but you should at least understand that it is
              plausible that the network should be able to. After all, neural
              networks learn by adjusting their parameters during training, and
              the function that maps words to embeddings can certainly be
              parameterized. For now, let's assume that the model associates
              with each word, a d<sub>model</sub>
              -dimensional vector. For concreteness, let's keep the value d
              <sub>model</sub>=512 in mind for now. Moreover, the mapping from
              words to embeddings must support tokens which correspond to
              punctuation, as well as an end-of-sentence token, which by
              convention is abbreviated as &lt;EOS&gt;. Such a token is
              necessary because, when feeding a sentence as input to the model,
              we must clarify when the sentence ends; conversely, when
              translating a sentence, we should require that our model tell us
              when the sentence ends. For our toy example, the embeddings might
              look something like:
            </p>
            <BlockEq
              text="$$
                \begin{aligned}
                  I &\to [0.32, 1.53, \dots, 0.96] \text{ (a 512-dimensional vector)}\\
                  made &\to [0.52, 2.01, \dots, 7.512]\\
                  up &\to [4.40, 8.57, \dots, 0.23]\\
                  a &\to [1.49, 2.34, \dots, 0.05]\\
                  story &\to [1.20, 8.33, \dots, 3.15]\\
                  . &\to [0.04, 0.01, \dots, 2.31]\\
                  \text{<EOS>} &\to [0.52, 1.34, \dots, 0.76]\\
                \end{aligned}
              $$"
              displayMode={true}
            />
            <p>
              Now, back to this nebulous concept of "attention" we have
              developed. As mentioned previously, for each word in the sentence,
              our model should be able to determine which other words it should
              pay attention to, as well as the effect of that other word on
              translation. Let's begin with the former: how does the model
              determine which other words a specified word should pay attention
              to? Since our model represents words using vectors, it should, for
              a given vector, be able to ask another vector the question, "to
              what extent should I pay attention to you?," and in turn receive a
              number in response: a very large number would indicate "paying
              attention to me is vital to translate yourself," and a very small
              number would indicate "I have no impact whatsoever on your
              translation." So, we need some function that takes two vectors and
              outputs a single number. If you've done any linear algebra, you'll
              immediately recognize such an operation as a dot product. Why
              don't we have the model learn a mapping from words to embeddings
              such that the dot product produces a number indicating the
              relative attention. Since the dot product attains larger values
              for vectors pointing in the same direction, this is tantamount to
              the idea that vectors pointing in the same direction should pay
              attention to each other. This approach almost works, but there's
              one small issue: the dot product operation is commutative. In
              other words, for two word embeddings a and b,
            </p>
            <BlockEq
              text="$$
          a \cdot b = b \cdot a
          $$"
              displayMode={true}
            />
            <p>
              See the issue? For concreteness, suppose "a" is the vector
              representing the word "I," while "b" is the vector representing
              the word "made." The implication of using a dot product to compute
              an attention score is that "I" should pay as much attention to
              "made" as "made" should to "I," which is not in general true. You
              might imagine, for example, that "made" should pay attention to
              "I," because the conjugation of the verb depends on the pronoun,
              but "I" doesn't care too much about being followed by the word
              "made" for the purposes of translation, because it will
              nonetheless be translated as "yo."
            </p>
            <p>
              We might be a bit disheartened by the realization that our nifty
              dot-product solution didn't work, but the issue is really quite
              easy to fix: we need separate embeddings to represent both the
              asker and answerer of the question "how much should I pay
              attention to you?" We'll call the embeddings corresponding with
              the asker "queries" and embeddings corresponding with the answerer
              "keys." This terminology is a bit strange, but the idea is that a
              query corresponds with asking a question (in this case, "how much
              should I pay attention to you?"), and the key is the information
              necessary to answer this question. In particular, taking the dot
              product of one word's query and another word's key provides the
              answer to the question. We're still using a dot product, but we've
              fixed the issue of commutativity by separating word embeddings
              into queries and keys. To find how much attention the word "I"
              should pay to the word "made," we take the dot product of "I"'s
              query and "made"'s key; to find how much attention the word "made"
              should pay to the word "I," we take the dot product of "made"'s
              query and "I"'s key. These dot products are <em>not</em> in
              general the same.
            </p>
            <p>
              So, how do we actually compute these queries and keys? Just as
              with the word embeddings, we'll let our model do the heavy
              lifting. For each word embedding, we'll apply two linear
              transformations (if you don't have a background in linear algebra,
              just think of a linear transformation as a special kind of
              function), the parameters of which can be learned during training.
              The result of applying the first linear transformation is the
              word's query, and the second is the word's key. So, we've
              specified how to compute the relative attention one word should
              pay to another, but how exactly can we use this information? Well,
              recall that our motivation for attention involved two crucial
              questions: "how much should I pay attention to you?" and "what is
              your implication on my translation?". The first question was
              resolved by mapping each word embedding to a query and key. To
              answer the second question, we will need to map each word to a
              third embedding, this time called a "value." Just like the query
              and the key, to obtain the value, we'll simply use a third linear
              transformation. The "value" can be thought of as representing the
              impact of one word on another in translation.
            </p>
            <ArticleImage src={QKVDiagram} width="70%" />
            <p>
              One more observation we should interject here is that we don't
              really want to translate a sequence of English words{" "}
              <em>directly</em> to a sequence of Spanish words, because the
              lengths of the sentences might be different in English and
              Spanish. In our example, "I made up a story" has five words, while
              "Inventé una historia" has three. Thus, we shouldn't think of our
              process of translation as simply translating each English word
              directly to Spanish. To get around this complication, as well as
              for several other reasons, the transformer employs an
              "encoder-decoder" architecture, which is a fancy way of saying
              that the English sentence is translated into an "intermediate
              representation," which consists of <em>context-aware</em> word
              embeddings, and then that intermediate representation is
              translated to Spanish. In the encoder, the English words pay
              attention to other English words to turn themselves into
              context-aware embeddings. In the decoder, the context-aware
              embeddings determine which Spanish words are included in the
              translation.
            </p>
            <p>
              Putting it all together, to create context-aware embeddings for a
              given word (say, "made"), we take a dot product of that word's
              query with each other word's value, which yields an attention
              score for each word (including the word "made"). We then multiply
              each word's value by its attention score, and sum up the results.
              The result is a context-aware embedding that contains some sort of
              "semantic essence" of the word "made." For example, this
              context-aware embedding likely includes the information that this
              "made" is associated with a future word "up," and thus refers to
              the act of fabrication.
            </p>
            <ArticleImage src={AttentionDiagram} />
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
            <p>
              Now that we have the basics of the attention mechanism nailed
              down, it's time to develop the mathematical formalism. Here, this
              just means packing all of our vectors neatly into matrices to
              cleanly represent the computation that maps embeddings to
              context-aware embeddings. For example, you'll notice that to
              compute the attention score for each pair of words, we need to
              take a dot product between each combination of two word emeddings.
              You might recognize such a computation from linear algebra as a
              matrix multiplication. In particular, let's form a query matrix{" "}
              <Eq text={"$Q$"} /> by packing each word's query together into a
              single matrix. Similarly, we'll form a key matrix{" "}
              <Eq text={"$K$"} /> by packing each word's key together into a
              single matrix. Then, for our seven-token input sentence:
            </p>
            <BlockEq
              text="$$
                Q = \begin{bmatrix} 
                  Q_{1,1} & Q_{1,2} & \cdots & Q_{1, 512} \\
                  Q_{2,1} & Q_{2,2} & \cdots & Q_{2, 512} \\
                  \vdots  & \vdots  & \ddots & \vdots     \\
                  Q_{7,1} & Q_{7,2} & \cdots & Q_{7, 512} \\
                \end{bmatrix}

                \begin{matrix}
                  &\text{Query (I)}, &d = 512\\
                  &\text{Query (Made)}, &d = 512\\
                  &\vdots\\
                  &\text{Query (<EOS>)}, &d = 512\\
                \end{matrix}
              $$"
              displayMode={true}
            />
            <BlockEq
              text="$$
                K = \begin{bmatrix} 
                  K_{1,1} & K_{1,2} & \cdots & K_{1, 512} \\
                  K_{2,1} & K_{2,2} & \cdots & K_{2, 512} \\
                  \vdots  & \vdots  & \ddots & \vdots     \\
                  K_{7,1} & K_{7,2} & \vdots & K_{7, 512} \\
                \end{bmatrix}

                \begin{matrix}
                  &\text{Key (I)}, &d = 512\\
                  &\text{Key (Made)}, &d = 512\\
                  &\vdots\\
                  &\text{Key (<EOS>)}, &d = 512\\
                \end{matrix}
              $$"
              displayMode={true}
            />
            <p>
              Looking at these matrices, you'll notice that taking a dot product
              between the <Eq text={"$i^\\text{th}$"} /> row of{" "}
              <Eq text={"$Q$"} /> and the <Eq text={"$j^\\text{th}$"} /> row of{" "}
              <Eq text={"$K$"} /> would yield the relative attention the{" "}
              <Eq text={"$i^\\text{th}$"} /> word should pay to the{" "}
              <Eq text={"$j^\\text{th}$"} /> word. How does this observation
              help us turn our computation into a matrix multiplication? Well,
              recall that when we multiply two matrices, each entry in the
              product is formed by taking a dot product of a row in the first
              matrix with a column in the second. Thus, if we take the transpose
              of the second matrix, its rows become columns, so multiplying{" "}
              <Eq text={"$Q$"} /> with the transpose of <Eq text={"$K$"} />{" "}
              involves taking exactly the dot products we're interested in. More
              succinctly,
            </p>
            <BlockEq
              text="$$
                QK^T = A
              $$"
              displayMode={true}
            />
            <p>
              where <Eq text={"$A$"} /> is a matrix we'll refer to as the
              attention matrix:
            </p>
            <BlockEq
              text="$$
                A = \begin{bmatrix} 
                  A_{1,1} & A_{1,2} & \cdots & A_{1, 7} \\
                  A_{2,1} & A_{2,2} & \cdots & A_{2, 7} \\
                  \vdots  & \vdots  & \ddots & \vdots     \\
                  A_{7,1} & A_{7,2} & \vdots & A_{7, 7} \\
                \end{bmatrix}

                \begin{matrix}
                  &\text{Attn. (I to each word)}, &d = 7\\
                  &\text{Attn. (Made to each word)}, &d = 7\\
                  &\vdots\\
                  &\text{Attn. (<EOS> to each word)}, &d = 7\\
                \end{matrix}
              $$"
              displayMode={true}
            />
            <p>
              In this matrix, <Eq text={"$A_{ij}$"} /> is the relative attention
              the <Eq text={"$i^\\text{th}$"} /> word should pay to the{" "}
              <Eq text={"$j^\\text{th}$"} />. Notice that this matrix
              multiplication represents not just the computation of the
              attention one word in the sentence should pay to each other word,
              but the attention <em>each</em> word in the sentence should pay to
              each other word (including itself). Pretty neat, and a lot more
              concise than that diagram in the previous section!
            </p>
            <p>
              The final step to arrive at our context-aware embeddings is to
              incorporate the values. As a refresher, let's quickly go over how
              we planned to use the values to produce context-aware embeddings.
              Let's suppose we're computing the context-aware embedding for the{" "}
              <Eq text={"$k^\\text{th}$"} /> word in the sentence (I'm using{" "}
              <Eq text={"$k$"} /> here instead of <Eq text={"$i$"} /> to clarify
              that we're focusing on one particular word for this example). We
              would first need to grab all of the attention scores{" "}
              <Eq text={"$A_{kj}$"} />, which correspond with the relative
              attention the <Eq text={"$k^\\text{th}$"} /> word needs to pay to
              each of the words in the input sentence. Then, for each of those
              words, we should multiply the attention score by the word's value
              vector, and sum these all up to produce the context-aware
              embedding for the <Eq text={"$k^\\text{th}$"} /> word. We would
              need to repeat this process for each word in the input sentence to
              compute all of the context-aware embeddings. To represent this
              computation neatly, we'll first need to introduce a matrix{" "}
              <Eq text={"$V$"} /> that contains the values for each input word,
              just as we did for the keys and queries:
            </p>
            <BlockEq
              text="$$
                V = \begin{bmatrix} 
                  V_{1,1} & V_{1,2} & \cdots & V_{1, 512} \\
                  V_{2,1} & V_{2,2} & \cdots & V_{2, 512} \\
                  \vdots  & \vdots  & \ddots & \vdots     \\
                  V_{7,1} & V_{7,2} & \vdots & V_{7, 512} \\
                \end{bmatrix}

                \begin{matrix}
                  &\text{Value (I)}, &d = 512\\
                  &\text{Value (Made)}, &d = 512\\
                  &\vdots\\
                  &\text{Value (<EOS>)}, &d = 512\\
                \end{matrix}
              $$"
              displayMode={true}
            />
            <p>
              Now, consider what happens when take a dot product of the first
              row of the attention matrix with the first column of the value
              matrix. Well, the first row of the attention matrix represents the
              relative attention "I" should pay to each word in the sentence.
              The first column in the value matrix is the first dimension of the
              value vector for each word. Since producing the context-aware
              embedding involves multiplying each word's attention score with
              its corresponding value vector and summing these products, the dot
              product gives the first dimension of the context-aware embedding
              for the first word. Similarly, suppose we were to take a dot
              product of the first row of the attention matrix with the second
              column of the value matrix. This time, since we're taking the
              second dimension of each value, we would get the second dimension
              of the context-aware embedding. Therefore, if we were to perform a
              matrix multiplication between the attention matrix and the value
              matrix, since we would be taking exactly these dot products
              between the first row of the attention matrix and columns of the
              value matrix, the first row of the product matrix would be
              precisely the first word's context-aware embedding. Through
              exactly the same process, the second row of the product matrix
              would give the second word's context-aware embedding, and so on.
              Therefore, we can multiply the attention matrix with the value
              matrix to obtain a matrix containing each of the context-aware
              embeddings! We're almost done here, but there are two small
              details that we need to include to match the paper's
              implementation.
              <ol>
                <li>
                  First, before multiplying the attention matrix with the value
                  matrix, the paper takes applies a softmax function to the
                  attention scores in the matrix, which, for each query,
                  normalizes the attention scores of each word in the sentence
                  so that they add up to 1. The paper doesn't include a proper
                  justification for its inclusion, but we can probably assume it
                  was empirically determined to improve the performance of the
                  Transformer. One way you might convince yourself that the
                  inclusion of this softmax is sensible is by considering what
                  might happen if one word in the sentence should pay virtually
                  no attention to another during the translation task. The
                  result of the dot product between the first word's query and
                  the second's key should intuitively be a large negative
                  number, but that would mean a large negative vector is
                  subtracted from the first word's context-aware embedding, just
                  because this word (which, remember, should have no effect on
                  translation!) was included. Intuitively, it makes a lot more
                  sense for attention scores to be numbers between 0 and 1,
                  which is exactly the effect of the softmax on the attention
                  matrix.
                </li>
                <li>
                  Second, the paper divides the attention scores by{" "}
                  <Eq text={"$\\sqrt{d_k}$"} /> before applying the softmax
                  function, where <Eq text={"$d_k$"} /> refers to the dimension
                  of the keys, which in our example is 512. You probably
                  shouldn't worry too much about this term: in fact, the paper's
                  tone suggests that they were quite uncertain about its effect,
                  even though they found it improved the performance. I'll just
                  quote them here:
                  <Blockquote cite="–Attention is All You Need (2017)">
                    We suspect that for large values of dk, the dot products
                    grow large in magnitude, pushing the softmax function into
                    regions where it has extremely small gradients. To
                    counteract this effect, we scale the dot products by{" "}
                    <Eq text={"$1/{\\sqrt{d_k}}$"} />.
                  </Blockquote>
                  This term is why the form of attention in the Transformer is
                  referred to as "<em>scaled</em> dot-product attention":
                  because the dot products between the queries and keys are
                  subsequently scaled.
                </li>
              </ol>
            </p>
            <p>
              Now, at last, we've arrived at the paper's formalism for scaled
              dot-product attention. Putting everything together, we can express
              the computation of the context-aware embedding matrix, referred to
              the paper as the "attention" as:
            </p>
            <BlockEq
              text="$$
                \text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V
              $$"
              displayMode={true}
            />
            <p>
              The word "attention" on the left side of the equation is a bit
              strange, and I would replace this word in your head with
              "context-aware embeddings" so that you don't confuse it with the
              matrix containing the attention scores. However, I think it's
              generally best to stick with the original paper's terminology to
              reduce friction in referencing the primary source, so we'll stick
              with this name throughout the article.
            </p>
            {/* <p>
              Advantages over previous models:
              <ol>
                <lem>Process whole sequence at once</lem>
                <lem>Explicitly facilitate long-range dependencies via attention</lem>
              </ol>
            </p> */}
          </div>
        </div>
      </div>
    </>
  );
};

export default TransformerPage1;

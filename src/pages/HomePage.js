import { useInView } from "react-intersection-observer";

import { Text, Grid } from "@mantine/core";

import ArticleCard from "../components/ArticleCard";

import QKVDiagram from "../images/QKVDiagram.svg";
import EncoderDiagram from "../images/EncoderDiagram.svg";
import DropoutThumbnail from "../images/DropoutThumbnail.svg";
import SkipConnectionThumbnail from "../images/SkipConnectionThumbnail.svg";
import AutoencoderThumbnail from "../images/AutoencoderThumbnail.svg";
import TTTIntroThumbnail from "../images/TTTIntroThumbnail.svg";

const HomePage = () => {
  const { ref: section1, inView: section1InView } = useInView({
    triggerOnce: true,
  });
  const { ref: section2, inView: section2InView } = useInView({
    triggerOnce: true,
  });
  const { ref: section3, inView: section3InView } = useInView({
    triggerOnce: true,
  });

  // const backpropagationCard = (
  //   <ArticleCard
  //     name="Backpropagation"
  //     // description="This is an excellent place to start if you're new to machine learning; backpropagation is at the foundation of all deep learning architectures. This article will delve into the theory behind why backpropagation allows a neural network to learn, and we'll nail in the concept with a practical example of digit recognition using PyTorch."
  //     description="How do networks update their weights to improve predictions?"
  //     comingSoon="true"
  //     img={QKVDiagram}
  //     pagePath="/coming-soon"
  //   />
  // );

  const dropoutCard = (
    <ArticleCard
      name="Dropout"
      description="Learn about how dropout layers help prevent overfitting in neural networks."
      img={DropoutThumbnail}
      pagePath="/dropout"
    />
  );

  const skipConnectionsCard = (
    <ArticleCard
      name="Skip Connections"
      description="Why do deeper neural networks sometimes perform worse, and how can we fix that?"
      img={SkipConnectionThumbnail}
      pagePath="/skip-connections"
    />
  );

  const autoencoderCard = (
    <ArticleCard
      name="Autoencoders"
      description="How can a neural network learn useful features from unlabeled training data?"
      img={AutoencoderThumbnail}
      pagePath="/autoencoders"
    />
  );

  const tttIntroCard = (
    <ArticleCard
      name="Setup and Introduction"
      description="We'll implement tic-tac-toe in Python as a starting point for developing capable AI for games."
      img={TTTIntroThumbnail}
      pagePath="/ttt-intro"
    />
  );

  const minimaxCard = (
    <ArticleCard
      name="Minimax and Alpha-Beta Pruning"
      description="How can we implement an algorithm that achieves perfect play in tic-tac-toe?"
      img={TTTIntroThumbnail}
      pagePath="/minimax"
    />
  );

  const mctsCard = (
    <ArticleCard
      name="Monte Carlo Tree Search"
      description="We discuss and implement MCTS, an algorithm capable of approximating perfect play by continuously improving its predictions."
      img={TTTIntroThumbnail}
      pagePath="/mcts"
    />
  );

  const transformerCard1 = (
    <ArticleCard
      name="The Transformer, Part 1"
      description="How does scaled dot-product attention accomodate long-range dependencies in sequences?"
      img={QKVDiagram}
      fit="cover"
      pagePath="/transformer1"
    />
  );

  const transformerCard2 = (
    <ArticleCard
      name="The Transformer, Part 2"
      description="We provide a treatment of multi-head attention and positional encoding to finish describing the encoder of the Transformer."
      img={EncoderDiagram}
      pagePath="/transformer2"
    />
  );

  const transformerCard3 = (
    <ArticleCard
      name="The Transformer, Part 3"
      description="We describe the architecture of the Transformer's decoder and finalize the details of the the model."
      inDevelopment="true"
      img={QKVDiagram}
      pagePath="/transformer3"
    />
  );

  return (
    <div style={{ marginLeft: "20%", marginRight: "20%" }}>
      <Text
        component="span"
        weight={700}
        style={{ fontSize: 60, lineHeight: "1em" }}
        margin="10px"
      >
        Machine learning concepts, <br />
      </Text>
      <Text
        component="span"
        // align="center"
        variant="gradient"
        weight={700}
        style={{ fontSize: 60 }}
      >
        explained intuitively.
      </Text>
      <br />
      <br />
      <Text style={{ fontSize: 21 }}>
        Finding high-quality machine learning resources shouldn't be an ordeal.
        ML Toolkit is an educational resource that eschews both the information
        density of technical research papers and the inconsistent explanations
        of many learning resources to provide a no-compromises means of
        understanding complex but powerful concepts.
      </Text>
      <br />
      <br />
      <br />
      <br />
      <div ref={section1}>
        <Text
          style={{ fontSize: 30 }}
          weight={600}
          mb="1rem"
          className={section1InView ? "fadeInText" : "hidden"}
        >
          Deep Learning Techniques
        </Text>
        <Text mb="1rem" className={section1InView ? "fadeFromLeft" : "hidden"}>
          What do deep learning researchers mean by "regularization"? Should you
          be adding dropout layers to your neural network? What is the
          difference between batch normalization and layer normalization? Why
          have skip connections become so prevalent? We'll take a deep dive into
          several techniques used in constructing and training neural networks,
          with hands-on experiments in PyTorch.
        </Text>
        <br />
        <Grid>
          <Grid.Col
            span={4}
            className={section1InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {dropoutCard}
          </Grid.Col>
          <Grid.Col
            span={4}
            className={section1InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {skipConnectionsCard}
          </Grid.Col>
          <Grid.Col
            span={4}
            className={section1InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {autoencoderCard}
          </Grid.Col>
        </Grid>
      </div>
      <br />
      <br />
      <br />
      <div ref={section2}>
        <Text
          style={{ fontSize: 30 }}
          weight={600}
          mb="1rem"
          className={section2InView ? "fadeInText" : "hidden"}
        >
          Games and Reinforcement Learning
        </Text>
        <Text mb="1rem" className={section2InView ? "fadeFromLeft" : "hidden"}>
          In 2016, DeepMind's AlphaGo became the first computer program to
          defeat a 9-dan professional Go player, a feat which was at the time
          thought to be a decade or more away. To understand both why achieving
          superhuman Go performance was understood to be such a herculean task,
          and how AlphaGo managed to crack it, we'll start from the basics of
          writing AI for games of perfect information, and we'll work our way
          toward replicating the architecture behind AlphaGo.
        </Text>
        <br />
        <Grid>
          <Grid.Col
            span={4}
            className={section2InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {tttIntroCard}
          </Grid.Col>
          <Grid.Col
            span={4}
            className={section2InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {minimaxCard}
          </Grid.Col>
          <Grid.Col
            span={4}
            className={section2InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {mctsCard}
          </Grid.Col>
        </Grid>
      </div>
      {/* <div ref={section2}> */}
      {/* <Text
          style={{ fontSize: 30 }}
          weight={600}
          mb="1rem"
          className={section1InView ? "fadeInText" : "hidden"}
        >
          Introduction to Deep Learning
        </Text>
        <Text mb="1rem" className={section1InView ? "fadeFromLeft" : "hidden"}>
          New to deep learning? This series will take you through fundamental
          concepts such as feedforward neural networks, activation functions,
          and backpropagation. You'll come out with an intuitive understanding
          of how a neural network can be trained to solve complex tasks and will
          be well-equipped to move onto more advanced concepts like
          convolutional neural networks.
        </Text> */}
      {/* <Text
          style={{ fontSize: 30 }}
          weight={600}
          mb="1rem"
          className={section2InView ? "fadeInText" : "hidden"}
        >
          Convolutional Neural Networks
        </Text>
        <Text mb="1rem" className={section2InView ? "fadeFromLeft" : "hidden"}>
          Why have convolutional neural networks become so prevalent in image
          recognition? This series will explain the theory behind what makes
          CNNs well-suited to image-based tasks, walk you through practice
          problems to reinforce your understanding, and provide a hands-on
          implementation exercise in PyTorch to compare the performance of CNNs
          with MLPs on a simple optical character recognition task.
        </Text> */}
      {/* <br /> */}
      {/* <Grid>
          <Grid.Col span={4} className={section1InView ? "fadeFromLeft delay1" : "hidden"}>{backpropagationCard}</Grid.Col>
          <Grid.Col span={4} className={section1InView ? "fadeFromLeft delay2" : "hidden"}>{backpropagationCard}</Grid.Col>
          <Grid.Col span={4} className={section1InView ? "fadeFromLeft delay3" : "hidden"}>{backpropagationCard}</Grid.Col>
        </Grid> */}
      {/* </div> */}
      <br />
      <br />
      <br />
      <div ref={section3}>
        <Text
          style={{ fontSize: 30 }}
          weight={600}
          mb="1rem"
          className={section3InView ? "fadeInText" : "hidden"}
        >
          The Transformer Architecture
        </Text>
        <Text mb="1rem" className={section3InView ? "fadeFromLeft" : "hidden"}>
          Already familiar with concepts in deep learning? This series provides
          an in-depth look into the Transformer, an attention-driven sequence
          transduction model. One of the most important deep learning
          architectures right now, the Transformer model has yielded
          state-of-the-art machine translation performance and drives models
          like GPT-3.
        </Text>
        <br />
        <Grid>
          <Grid.Col
            span={4}
            className={section3InView ? "fadeFromLeft delay1" : "hidden"}
          >
            {transformerCard1}
          </Grid.Col>
          <Grid.Col
            span={4}
            className={section3InView ? "fadeFromLeft delay2" : "hidden"}
          >
            {transformerCard2}
          </Grid.Col>
          <Grid.Col
            span={4}
            className={section3InView ? "fadeFromLeft delay3" : "hidden"}
          >
            {transformerCard3}
          </Grid.Col>
        </Grid>
      </div>
    </div>
  );
};

export default HomePage;

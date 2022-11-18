import { Text, Grid } from "@mantine/core";
import ArticleCard from "../components/ArticleCard";

const HomePage = () => {
  const backpropagationCard = (
    <ArticleCard
      name="Backpropagation"
      // description="This is an excellent place to start if you're new to machine learning; backpropagation is at the foundation of all deep learning architectures. This article will delve into the theory behind why backpropagation allows a neural network to learn, and we'll nail in the concept with a practical example of digit recognition using PyTorch."
      description="How do networks update their weights to improve predictions?"
      comingSoon="true"
      img="https://upload.wikimedia.org/wikipedia/commons/6/60/ArtificialNeuronModel_english.png"
      pagePath="/coming-soon"
    />
  );

  const transformerCard1 = (
    <ArticleCard
      name="The Transformer, Part 1"
      // description="Already familiar with concepts in deep learning? This article provides an in-depth look into the Transformer, an attention-driven sequence transduction model. Arguably the most important deep learning architecture right now, this architecture has yielded state-of-the-art machine translation performance and drives models like GPT-3."
      description="How does scaled dot-product attention accomodate long-range dependencies in sequences?"
      img="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1252%2F1*JuGZaZcRtmrtCEPY8qfsUw.png&f=1&nofb=1&ipt=af10833e0a6118828768c9d77f23d5f35720aa351d59c2f28332c48f9ee97c8c&ipo=images"
      pagePath="/transformer"
    />
  );

  const transformerCard2 = (
    <ArticleCard
      name="The Transformer, Part 2"
      // description="Already familiar with concepts in deep learning? This article provides an in-depth look into the Transformer, an attention-driven sequence transduction model. Arguably the most important deep learning architecture right now, this architecture has yielded state-of-the-art machine translation performance and drives models like GPT-3."
      description="We provide a treatment of multi-headed attention and positional encoding to finish describing the encoder of the Transformer."
      inDevelopment="true"
      img="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1252%2F1*JuGZaZcRtmrtCEPY8qfsUw.png&f=1&nofb=1&ipt=af10833e0a6118828768c9d77f23d5f35720aa351d59c2f28332c48f9ee97c8c&ipo=images"
      pagePath="/transformer"
    />
  );

  const transformerCard3 = (
    <ArticleCard
      name="The Transformer, Part 3"
      // description="Already familiar with concepts in deep learning? This article provides an in-depth look into the Transformer, an attention-driven sequence transduction model. Arguably the most important deep learning architecture right now, this architecture has yielded state-of-the-art machine translation performance and drives models like GPT-3."
      description="We describe the architecture of the Transformer's decoder and finalize the details of the the model."
      comingSoon="true"
      img="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1252%2F1*JuGZaZcRtmrtCEPY8qfsUw.png&f=1&nofb=1&ipt=af10833e0a6118828768c9d77f23d5f35720aa351d59c2f28332c48f9ee97c8c&ipo=images"
      pagePath="/transformer"
    />
  );

  return (
    <div style={{ marginLeft: "15%", marginRight: "15%" }}>
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
      <Text size="xl">
        Building machine learning tools shouldn't be difficult. Gecko is a free
        educational resource that eschews both the technical language of
        research papers and the inconsistent explanations of most learning
        resources to provide a no-compromises means of understanding complex but
        powerful concepts.
      </Text>
      <br />
      <br />
      <br />
      {/* <Text style={{ fontSize: 40 }} weight={700} mb="1rem">
        Start Learning Now
      </Text> */}
      <br />
      <Text style={{ fontSize: 30 }} weight={600} mb="1rem">
        Introduction to Deep Learning
      </Text>
      <Text mb="1rem">
        New to deep learning? This series will take you through fundamental
        concepts such as feedforward neural networks, activation functions, and
        backpropagation. You'll come out with an intuitive understanding of how
        a neural network can be trained to solve complex tasks and will be
        well-equipped to move onto more advanced concepts like convolutional
        neural networks.
      </Text>
      <br />
      <Grid>
        <Grid.Col span={4}>{backpropagationCard}</Grid.Col>
        <Grid.Col span={4}>{backpropagationCard}</Grid.Col>
        <Grid.Col span={4}>{backpropagationCard}</Grid.Col>
      </Grid>
      <br />
      <br />
      <br />
      <Text style={{ fontSize: 30 }} weight={600} mb="1rem">
        The Transformer Architecture
      </Text>
      <Text mb="1rem">
        Already familiar with concepts in deep learning? This series provides an
        in-depth look into the Transformer, an attention-driven sequence
        transduction model. Arguably the most important deep learning
        architecture right now, this architecture has yielded state-of-the-art
        machine translation performance and drives models like GPT-3.
      </Text>
      <br />
      <Grid>
        <Grid.Col span={4}>{transformerCard1}</Grid.Col>
        <Grid.Col span={4}>{transformerCard2}</Grid.Col>
        <Grid.Col span={4}>{transformerCard3}</Grid.Col>
      </Grid>
    </div>
  );
};

export default HomePage;

import React from "react";
import { useRef } from "react";

import { Anchor, Code } from "@mantine/core";
import { Prism } from "@mantine/prism";

import Eq from "../components/Eq";
import BlockEq from "../components/BlockEq";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleImage from "../components/ArticleImage";

import mnist from "../images/mnist.png";
import mnist_color from "../images/mnist_color.png";
import dropout_experiment_graph from "../images/dropout_experiment_graph.png";
import AutoencoderExample from "../images/AutoencoderExample.svg";
import AutoencoderDiagram from "../images/AutoencoderDiagram.svg";

const AutoencoderPage = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Introduction",
      id: "introduction",
    },
    {
      name: "What Are Autoencoders?",
      id: "what",
    },
    {
      name: "Why Are They Useful?",
      id: "why",
    },
    {
      name: "How Do We Use Them?",
      id: "how",
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
            <ArticleTitle name={"Autoencoders"} />
            <ArticleSubtitle name={"Learning Useful Features Without Labels"} />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              Autoencoders are a versatile neural network architecture capable
              of learning efficient representations of input data without the
              use of labels. Because they are able to find useful features for a
              dataset without supervision, autoencoders can prove useful in
              tasks including clusterization, data compression, and anomaly
              detection. In this article, we'll motivate and describe the
              architecture of autoencoders, discuss several areas of
              application, and provide a concrete example of how an autoencoder
              can be implemented in PyTorch to detect anomalous samples in a
              dataset.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              To train a neural network without labels, we need to come up with
              a "task" that the network can solve using only the input data.
              Perhaps the most obvious task one might devise is to ask the
              neural network to reproduce its own input, which could be realized
              by computing the loss as the mean squared error between the input
              sample <Eq text="$x_i$" /> and the output{" "}
              <Eq text="$\hat{y_i}$" /> of the network.
              <BlockEq
                text="$$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i}) = {\frac{1}{n}} \sum_{i=1}^n (x_i - \hat{y_i})$$"
                displayMode={true}
              />
              Of course, this task is trivial--the identity function attains a
              loss of zero on any input sample--and there isn't really anything
              for the network to learn. Now, however, imagine that this neural
              network contains a hidden layer whose dimension is smaller than
              that of the input layer.
            </p>
            <ArticleImage src={AutoencoderExample} width="30%" />
            <p>
              It is no longer possible for the neural network to learn the
              identity function, because the input cannot simply be copied to
              the lower-dimensional hidden layer. Instead, the neural network
              must now effectively learn to encode its input in the hidden layer
              in a way that allows itself to subsequently reconstruct the input
              from the compressed representation. The task we've devised has
              gone from trivial, to somewhat challenging. Such a neural network,
              which learns by itself to encode input samples into a
              lower-dimensional space, is referred to as an "autoencoder."
            </p>
            <p>
              While the autoencoder in the diagram above depicts a multi-layer
              perceptron, autoencoders can come in a variety of forms. For
              example, a convolutional autoencoder could be built by using
              convolutional layers instead of, or in addition to the fully
              connected layers. In general, we describe autoencoders as
              consisting of an "encoder" and a "decoder," where the role of the
              encoder is to produce a low-dimensional representation of the
              input data, and the role of the decoder is to use this
              representation to reconstruct the image. The lowest-dimensional
              hidden layer in the autoencoder is sometimes referred to as the
              "bottleneck."
            </p>
            <ArticleImage src={AutoencoderDiagram} width="60%" />
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              So, we've come up with a non-trivial unsupervised task for our
              neural network to solve, but how can we actually put what the
              autoencoder learns to use? One observation we might make is that
              since the autoencoder must use the information in the bottleneck
              to reconstruct the input, it should learn automatically to come up
              with useful features for the dataset on which it is trained.
              Indeed, if the encoder produces more informative features at the
              bottleneck, the decoder should be able to use these features to
              produce more accurate reconstructions of the input. The bottleneck
              can thus be thought of as containing a vector in a "latent space,"
              or a multi-dimensional space of features, and we expect that these
              features should be a good representation of the input data. The
              ability to learn useful features on an unlabelled dataset is quite
              powerful, and these features could be used for a wide array of
              tasks.
            </p>
            <p>
              We could, for example, use these features directly to visualize a
              dataset by plotting the latent space representations of input data
              produced by the encoder. If there are several distinct classes of
              samples in the dataset, samples from the same class will likely
              produce a similar representation in the latent space, so we might
              gain insight about the nature of the dataset by clusterizing
              vectors in the latent space.
            </p>
            <p>
              Alternatively, these features could be used for a "downstream"
              task, such as classification. You might imagine that we have an
              unlabelled dataset for which it would be costly to hand-label each
              input sample. Since fully supervised training isn't an option, we
              might instead opt to label a small percentage of the dataset. We
              could train an autoencoder on the unlabelled data, and train a
              neural network on top of the encoder to classify samples using the
              labelled data. This paradigm, in which machine learning is
              conducted using a large amount of unlabelled data and a small
              amount of labelled data, is sometimes known as "semi-supervised
              learning."
            </p>
            <p>
              Perhaps the most obvious application of autoencoders is data
              compression; after all, the task on which they are trained is to
              efficiently compress and reconstruct data. After an autoencoder is
              trained on a dataset, we could use the encoder portion of the
              network to compress data of a similar nature. The compressed data
              could then, for example, be stored or sent across a network. To
              recover the data, we would feed the compressed data to the decoder
              portion of the network to provide an accurate (but lossy)
              reconstruction of the original data. Compression can help us save
              on storage space or communication costs.
            </p>
            <p>
              One final application of autoencoders I'll mention here is anomaly
              detection. Anomalous samples can be thought of as those samples
              which fall outside the typical distribution of training data. An
              anomalous sample in a dataset of images of manufactured goods
              might indicate a defect in a product; an anomaly in a dataset of
              audio recordings of bird calls might be a bird species that hasn't
              yet been observed. Autoencoders can be used to detect anomalies by
              monitoring value of the loss function, also known as the
              reconstruction error, for each input sample. Since the autoencoder
              has been trained on samples within the distribution of the
              training data, we might expect that it should attain better
              performance, and thus lower reconstruction error, on samples that
              lie within this distribution. When presented with an anomalous,
              out-of-distribution sample, the autoencoder's implicit assumptions
              about the input could fail, and we might anticipate a larger
              reconstruction error. Therefore, autoencoders can be used to
              detect anomalies in a dataset.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
          </div>
        </div>
      </div>
    </>
  );
};

export default AutoencoderPage;

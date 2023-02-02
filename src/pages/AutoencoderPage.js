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

import autoencoder_loss from "../images/autoencoder_loss.png";
import autoencoder_anomalies from "../images/autoencoder_anomalies.png";
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
            <p>
              To demonstrate how an autoencoder might be used on a practical
              task, we'll implement an autoencoder that can detect anomalies in
              the MNIST dataset of handwritten digits. It should be noted that
              if we were to implement an autoencoder for an anomaly detection
              task on a real dataset, we'd likely want to devise some way of
              validating our model after it is trained, such as by ensuring that
              it flags known anomalous samples. For this simple example,
              however, we'll just perform a crude qualitative examination of the
              samples with the <Eq text="$$k$$" /> largest reconstruction
              errors, which should be enough to convince you that autoencoders
              can be a viable strategy for solving anomaly detection tasks.
            </p>
            <p>
              The code I wrote for this exercise is built off the MNIST code
              from the{" "}
              <Anchor
                href="https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html"
                target="_blank"
              >
                PyTorch Quickstart Guide
              </Anchor>
              . First, if you haven't installed PyTorch on your local machine,
              run:
            </p>
            <Prism language="python">
              {`pip install torch==1.9.0 torchvision==0.10.0`}
            </Prism>
            <p>
              We'll start off by importing the necessary utilities from PyTorch,
              as well as matplotlib.pyplot, which will allow us to plot our
              results.
            </p>
            <Prism language="python">
              {`import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt
`}
            </Prism>
            <p>
              Our autoencoder will consist of an encoder that flattens the input
              image and reduces the dimension from an input dimension of 784 to
              a bottleneck dimension of 50, using a few hidden layers with ReLU
              activation. The decoder will use identical hidden layers, with the
              dimensions reversed, to bring the input back up to a dimension of
              784, with a sigmoid activation function at the output layer
              ensuring the pixel values of the output image stay between 0 and
              1. We'll also unflatten the output of the autoencoder to match the
              dimensions of the original image.
            </p>
            <Prism language="python">
              {`class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )
        self.unflatten = nn.Unflatten(1, (1, 28, 28))

    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.unflatten(x)
        return x`}
            </Prism>
            <p>
              We define functions <Code>train()</Code> and <Code>test()</Code>{" "}
              almost exactly as they're defined in the PyTorch Quickstart Guide.
              However, instead of computing the loss using the model output and
              the sample's label, we compute the reconstruction error using the
              model's output and the original sample.
            </p>
            <Prism language="python">
              {`def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, X).item()
    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \\n")
    return test_loss
`}
            </Prism>
            <p>
              Since we'd like to detect the samples that fall out of
              distribution, we'll implement a function{" "}
              <Code>detect_anomalies()</Code> to compute the reconstruction
              error for each sample in the test dataset and keep track of the
              original and reconstructed images for the samples with the five
              largest reconstruction errors.
            </p>
            <Prism language="python">
              {`def detect_anomalies(dataloader, model, loss_fn):
    model.eval()
    top_5_anomalies = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            for (orig, recon) in zip(X, pred):
                loss = loss_fn(orig, recon).item()
                top_5_anomalies.append((orig, recon, loss))
            top_5_anomalies = sorted(top_5_anomalies, key=(lambda a: a[2]), reverse=True)[:5]
    return top_5_anomalies`}
            </Prism>
            <p>
              Finally, we'll train our autoencoder, and plot its loss curve, as
              well as the top five anomalies it finds.
            </p>
            <Prism language="python">
              {`if __name__ == "__main__":
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 256

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = Autoencoder()

    model.to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    test_losses = []

    epochs = 80
    for t in range(epochs):
        print(f"Epoch {t+1}\\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        loss = test(test_dataloader, model, loss_fn)
        test_losses.append(loss)
    print("Done!")

    # Plot loss curve.
    plt.plot(test_losses)
    plt.xlabel("Training Epoch")
    plt.ylabel("Test Loss")
    plt.title("Autoencoder Test Loss on MNIST Dataset")
    plt.savefig(f"test_loss_plot")
    
    # Get top 5 anomalies and plot.
    top_5_anomalies = detect_anomalies(test_dataloader, model, loss_fn)
    fig, axs = plt.subplots(nrows=2, ncols=5)
    for i, (img, recon, _) in enumerate(top_5_anomalies):
        axs[0][i].imshow(img[0].cpu())
        axs[1][i].imshow(recon[0].cpu())
        axs[0][i].xaxis.set_visible(False)
        axs[0][i].yaxis.set_visible(False)
        axs[1][i].xaxis.set_visible(False)
        axs[1][i].yaxis.set_visible(False)
    plt.savefig("MNIST Anomalies")`}
            </Prism>
            <p>Below is the loss curve from the training run on my machine:</p>
            <ArticleImage src={autoencoder_loss} width="60%" />
            <p>
              I've also included the original and reconstructed images from the
              samples with the top 5 largest reconstruction errors.{" "}
            </p>
            <ArticleImage
              src={autoencoder_anomalies}
              width="80%"
              caption="The original (top) and reconstructed (bottom) images from the samples with the largest reconstruction errors."
            />
            <p>
              A quick qualitative examination suggests these samples could
              indeed be considered anomalous; two of the samples (the images of
              the 3 and the 2) have artifacts on the right hand side, two of the
              samples (the 0 and the 7) have unique strokes or are written in a
              unique way, and one of the samples (the 8) is cut off. While a
              real application might warrant a far more rigorous validation of
              the model, our quick-and-dirty experiments shows that autoencoders
              are at least a viable strategy for tackling the anomaly detection
              problem.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default AutoencoderPage;

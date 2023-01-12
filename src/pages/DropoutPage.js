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
import ArticleFeedback from "../components/ArticleFeedback";
import ArticleImage from "../components/ArticleImage";

import mnist from "../images/mnist.png";
import mnist_color from "../images/mnist_color.png";
import dropout_experiment_graph from "../images/dropout_experiment_graph.png";
import DropoutDiagram from "../images/DropoutDiagram.svg";

const DropoutPage = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Introduction",
      id: "introduction",
    },
    {
      name: "Overfitting",
      id: "overfitting",
    },
    {
      name: "Dropout",
      id: "dropout",
    },
    {
      name: "An Experiment",
      id: "experiment",
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
            <ArticleTitle name={"Dropout"} />
            <ArticleSubtitle
              name={"Why Omitting Neurons Can Prevent Overfitting"}
            />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              Dropout is a technique used in the training of deep neural
              networks that involves randomly removing, or "dropping out"
              neurons from layers of a neural network during training. It may
              initially seem unintuitive that the performance of a network could
              be improved by what is effectively reducing the number of features
              the network can use to make predictions. However, the method has a
              very concrete goal--to reduce overfitting on training data--and by
              first looking into the issue dropout aims to solve, it becomes
              almost trivially easy to understand the purpose of the technique.
            </p>
            <p>
              We'll start by discussing the problem of overfitting, and I'll
              provide a pathological example of how such a problem might
              manifest in a small training set. We'll use this discussion to
              motivate dropout as a potential solution, and we'll concretely
              define its implementation. Finally, we'll adapt an experiment from
              the original paper, "Improving neural networks by preventing
              co-adaptation of feature detectors" (Hinton et al. 2012), to get
              some hands-on time with the dropout technique in PyTorch, and so
              that you can convince yourself of its utility.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              In machine learning, we generally define a "training set"
              consisting of data used to train a model, and a "test set"
              containing data the model hasn't encountered before. The test set
              can be used to assess the validity of a trained model by checking
              whether what the model has learned can generalize. Overfitting
              occurs when a model picks up on patterns in the training data that
              are the result of noise, and it therefore struggles to generalize
              its knowledge to new data in the test set. A hallmark of
              overfitting would be the observation of far superior performance
              on training data, as compared with the test data. Overfitting
              might occur becuase a model is trained for so long on the training
              data that it begins to make use of features that don't generalize
              well to examples outside of the training set. A model is also more
              likely to overfit if the training set contains a relatively small
              number of samples, because there will be more "false patterns" for
              it to pick up on.
            </p>
            <p>
              To make this idea concrete, consider training a classifier on the
              MNIST dataset, which consists of images of the handwritten digits
              from 0 to 9. For simplicity, imagine we're only planning on
              training and testing the model on images of the digits 2 and 3.
              Now, as a pathological example, suppose our training set consisted
              only of three images of the digit 2 and three images of the digit
              3, as shown below.
            </p>
            <ArticleImage src={mnist} width="100%" />
            <p>
              A model trained on this training set would almost immediately be
              able to attain perfect accuracy, because it only needs to learn to
              correctly label these six images. Here's an example of one way the
              model might achieve this: simply look at the pixel at position
              (15, 15) from the top-left of the image. If the pixel is black,
              label the image "2," and if the pixel is white, label the image
              "3." To prove that such a function does indeed produce the correct
              labels for the six images, in the figure below, I've colored the
              pixels at position (15, 15) in green if they were originally black
              and in red if they were originally white.
            </p>
            <ArticleImage src={mnist_color} width="100%" />
            <p>
              Obviously, this model is problematic: relying on the specific
              value of a single pixel is not an effective way to distinguish
              between these digits, and the fact that a particular pixel was
              black in all of the images of "2" and white in all of the images
              of "3" can be attributed primarily to random chance. However,
              because our training set is so small, our model doesn't know any
              better than to rely on this pixel: it's as good of a feature for
              classification as any other. The model has essentially overfit on
              the training data, and its reliance on a particular pixel would
              inhibit its performance on a test set. Of course, it is
              unrealistic for a training set to contain only six images, but
              this example illustrates a more general issue: a model might pick
              up on features resulting from noise that help it perform well on
              the training data, but ones which are not useful in general.
              Here's where dropout comes in handy.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              In the dropout technique, during training, with some probability,
              each neuron in the input layer and the hidden layer may be
              removed, meaning it does not contribute to the next layer. The
              figure below illustrates what a simple multi-layer perceptron
              might look like with and without dropout.
            </p>
            <ArticleImage
              src={DropoutDiagram}
              width="65%"
              caption="On the left is a multi-layer perceptron neural network with no dropout, and on the right is a multi-layer perceptron with random dropout in the input layer and the hidden layer. The input layer has one neuron removed, and the hidden layer has two neurons removed; the removed neurons have their opacity reduced and their connections removed."
            />
            <p>
              It should be noted that dropout only occurs during training, so
              that the network can use all of the input data and features during
              inference. Moreover, during training, it is standard to rescale
              the outputs of the neurons that are not removed by a factor of{" "}
              <Eq text="$1/(1-p)$" />, where <Eq text="$p$" /> is the
              probability of dropout, in order to compensate for the weights of
              the neurons that are removed.
            </p>
            <p>
              With dropout, the neural network can no longer rely on the
              presence of particular neurons to fit its training data, so it
              must learn several useful features that are independent of each
              other. In the context of our example on the MNIST dataset, relying
              on the value of one particular pixel is no longer a viable
              strategy, because that pixel, or a neuron in a hidden layer
              corresponding with that feature, might be dropped out. The
              original paper on the dropout technique, "Improving neural
              networks by preventing co-adaptation of feature detectors" (Hinton
              et al. 2012) suggests that dropout can be thought of as a means of
              preventing "complex co-adaptations in which a feature detector is
              only helpful in the context of several other specific feature
              detectors." We can also think of our pathological example in the
              MNIST dataset as learning co-adapted features, or features that
              rely on each other. One neuron in the hidden layer of our
              classifier might take on a value based on the color of the pixel
              at position (15, 15), and the other neurons might learn to cancel
              each other out. The latter neurons are co-adapted with the former,
              because their cancellation is only reasonable in the context of
              the feature detector that looks at the color of the pixel at
              position (15, 15). If the network has to learn many features that
              don't depend on each other, it's much less likely that it will
              rely entirely on the effects of noise in the training data for
              making predictions. In short, dropout allows a neural network to
              learn better features which are more likely to generalize to new
              data.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
            <p>
              To illustrate the utility of incoporating dropout in the training
              of neural networks, we'll run an experiment in PyTorch adapted
              from one the experiments from the original paper. The example code
              is built off the MNIST code from the{" "}
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
            <Prism language="python" withLineNumbers>
              {`import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt`}
            </Prism>
            <p>
              One of the experiments the authors run uses the MNIST dataset to
              compare the accuracy of a neural network with and without dropout
              layers. To do this, they use a multi-layer perceptron consisting
              of an input layer containing 784 neurons (corresponding with the
              28 x 28 images of the digits), two hidden layers each containing
              800 neurons each, and an output layer containing 10 neurons
              (corresponding with the 10 labels for digits in the dataset).
              We'll follow the same structure, using ReLU activation functions
              after each linear layer, and using the built in{" "}
              <Code>nn.Dropout</Code> function for the dropout layers. As in the
              paper, we'll use a dropout probability of 0.2 for the input layer,
              and a dropout probability of 0.5 for each hidden layer.
            </p>
            <Prism language="python" withLineNumbers>
              {`class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class DropoutNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(28*28, 800),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits`}
            </Prism>
            <p>
              We define functions <Code>train()</Code> and <Code>test()</Code>{" "}
              just as they're defined in the PyTorch Quickstart Guide, with
              minor modifications in <Code>test()</Code> to return the accuracy,
              allowing us to track the performance of each model over time.
            </p>
            <Prism language="python" withLineNumbers>
              {`def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    acc = 100 * correct
    print(f"Test Error: \\n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \\n")
    return acc`}
            </Prism>
            <p>
              All that's left to do is to test each model! The original paper
              uses momentum and an optimized learning rate schedule, but here I
              use the Adam optimizer from <Code>torch.optim.Adam</Code> to keep
              things simple. The original paper trained for 3,000 epochs, but
              with this setup, you should be able to train each model for only
              50 epochs for the separation between the two models to become
              clear.
            </p>
            <Prism language="python" withLineNumbers>
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
    
    models = {
        "no_dropout": NeuralNetwork(),
        "dropout": DropoutNeuralNetwork()
    }
    for (model_name, model) in models.items():
        print(f"Evaluation {model_name} model.")

        model.to(device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        test_accs = []

        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t+1}\\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            acc = test(test_dataloader, model, loss_fn)
            test_accs.append(acc)
        print("Done!")

        plt.plot(test_accs, label=model_name)
    
    plt.xlabel("Training Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison for MLPs With and Without Dropout")
    plt.legend()
    plt.savefig(f"test_acc_plot")`}
            </Prism>
            <p>I've included the results of my training below:</p>
            <ArticleImage src={dropout_experiment_graph} width="60%" />
            <p>
              Notice that the performance of both models is roughly similar
              until the test accuracy reaches a certain point. After this, the
              model that doesn't use dropout overfits to the training data, and
              its performance on the test data isn't able to improve further.
              Meanwhile, the model with dropout is able to continue improving,
              achieving superior performance compared with the other model.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default DropoutPage;

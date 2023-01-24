import React from "react";
import { useRef } from "react";

import { Anchor, Code } from "@mantine/core";
import { Prism } from "@mantine/prism";

import Eq from "../components/Eq";

import ArticleNavigation from "../components/ArticleNavigation";
import ArticleHeader from "../components/ArticleHeader";
import ArticleTitle from "../components/ArticleTitle";
import ArticleSubtitle from "../components/ArticleSubtitle";
import ArticleAuthor from "../components/ArticleAuthor";
import ArticleImage from "../components/ArticleImage";

import SkipConnectionDiagram from "../images/SkipConnectionDiagram.svg";
import SkipConnectionDiagram2 from "../images/SkipConnectionDiagram2.svg";

import skip_connection_experiment_graph from "../images/skip_connection_experiment_graph.png";

const SkipConnectionPage = () => {
  const contentRef = useRef(null);

  const sectionHeaders = [
    {
      name: "Introduction",
      id: "introduction",
    },
    {
      name: "Why Skip Connections?",
      id: "skip-connections",
    },
    {
      name: "What About Dimensionality?",
      id: "dimensionality",
    },
    {
      name: "Experimental Comparison",
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
            <ArticleTitle name={"Skip Connections"} />
            <ArticleSubtitle
              name={
                "Improving the Performance of Deep Neural Networks by Hard-Coding the Identity Mapping"
              }
            />
            <ArticleAuthor />
            <ArticleHeader sectionHeader={sectionHeaders[0]} />
            <p>
              In 2015, teams of researchers competed to claim the top spot in
              the ILSVRC competition, which required achieving the lowest
              possible error on the ImageNet classification task. This task
              required labelling images from over 1,000 different classes, and
              evidence suggested that using convolutional neural networks with a
              high depth was a critical requirement for success. However,
              Microsoft researchers Kaiming He, Xiangyu Zhang, Shaoqing Ren, and
              Jian Sun noticed something strange: beyond a certain depth, the
              performance of deeper neural networks was actually worse than that
              of their shallower counterparts.
            </p>
            <p>
              To demonstrate this, the authors trained both a 20-layer and a
              56-layer deep neural network on an image classification task using
              the CIFAR-10 dataset. They observed that, during training, the
              error of the 56-layer network was consistently higher than that of
              the 20-layer network, which seems counterintuitive: since the
              deeper network emcompasses the shallower one, but simply with
              additional parameters available for tuning, shouldn't its
              performance be at least as good? Indeed, if the additional layers
              in the deeper network simply comprised the identity mapping, the
              performance would be equal to that of the shallower network. At
              first, the issue might be chalked up to overfitting, which might
              arise because a more complex model is more capable of learning a
              function that takes advantage of noise in the training set.
              However, this possibility can be discarded, because the training
              error, not just the test error, suffered when the networks grew
              deeper.
            </p>
            <p>
              In order to alleviate this issue, the researchers came up with an
              idea known as "skip connections." By incoporating these
              connections into a convolutional neural network, they formed a
              residual neural network, or ResNet, that earned them first place
              in the 2015 ILSVRC classification task. Let's find out how they
              did it, and why skip connections are so powerful.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[1]} />
            <p>
              The motivation for skip connections comes from the fact that
              deeper neural networks <em>should</em> be able to perform at least
              as well as their shallower counterparts. In their paper, He et al.
              point out that one could construct a deep neural network that
              performs just as well as the shallow one, just by copying the
              parameters of the shallow neural network, and tacking on more
              layers which corresponding with the identity mapping, meaning they
              perform no transformation on the data at all. When deeper neural
              networks were trained, however, the network did not learn this
              solution. The idea behind skip connections is to precondition the
              network by hard-coding the identity mapping as the default mapping
              from one layer to a future layer. In practice, this involves
              adding a vector <Eq text="$$\textbf{x}$$" /> from one part of a
              neural network to the output{" "}
              <Eq text="$$\mathcal{F}(\textbf{x})$$" /> of a future part of the
              network. The diagram below, adapted from the original paper,
              illustrates how a skip connection might look in a simple
              feedforward network. Note that the addition of the vector{" "}
              <Eq text="$$\textbf{x}$$" /> precedes the activation function of
              the layer to which it is added.
            </p>
            <ArticleImage src={SkipConnectionDiagram} />
            <p>
              He et al. formalize the concept as follows: if the optimal mapping
              from one layer for a future layer is{" "}
              <Eq text="$$\mathcal{H}(x)$$" />, the layers in between will learn
              the mapping{" "}
              <Eq text="$$\mathcal{F}(\textbf{x}) = \mathcal{H}(\textbf{x}) - \textbf{x}$$" />
              , and the identity mapping will be added to produce{" "}
              <Eq text="$$\mathcal{F}(x) + \textbf{x}$$" />. The paper
              conjectures and verifies experimentally that layers in a neural
              network have an easier time learning{" "}
              <Eq text="$$\mathcal{F}(\textbf{x})$$" /> than they do directly
              learning <Eq text="$$\mathcal{H}(\textbf{x})$$" />. With skip
              connections, a deep neural network can be thought of as by default
              copying over the solution found by its shallower counterpart. The
              layer itself can improve upon this solution by learning a function
              that is added to the identity mapping, which the authors refer to
              as a "residual mapping."
            </p>
            <p>
              Skip connections add no additional learnable parameters, and they
              don't increase the computational complexity of the neural network.
              Thus, they are a simple but highly effective way of improving the
              performance of deep neural networks.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[2]} />
            <p>
              Skip connections, as we've formulated them so far, can only work
              if the dimension of the input to the layer at which the connection
              begins is equal to the dimension of the output of the layer at
              which the connection ends. Indeed, addition is only defined for
              vectors of the same dimension. To allow skip connections to be
              applied even in situations where this is not the case, the authors
              of the paper specify and test a couple of ways to increase the
              dimensionality of the vector <Eq text="$$\textbf{x}$$" />. Using
              these methods, if <Eq text="$$\mathcal{F}(\textbf{x})$$" /> has a
              higher dimensionality than <Eq text="$$\textbf{x}$$" />, a skip
              connection can still be used.
            </p>
            <p>
              The first method is to pad <Eq text="$$\textbf{x}$$" /> with
              zeroes to make up the difference in dimensionality, and the second
              is to apply a linear projection to bring{" "}
              <Eq text="$$\textbf{x}$$" /> to a higher-dimensional space. Using
              the second method, instead of a stack of layers with a skip
              connection producing the output{" "}
              <Eq text="$$\mathcal{F}(\textbf{x}) + \textbf{x}$$" />, it would
              produce the output{" "}
              <Eq text="$$\mathcal{F}(\textbf{x}) + W_s\textbf{x}$$" />, where{" "}
              <Eq text="$$W_s$$" /> is a linear projection that matches the
              dimensionality of <Eq text="$$\textbf{x}$$" /> to that of{" "}
              <Eq text="$$\mathcal{F}(\textbf{x})$$" />. The two methods are
              illustrated below.
            </p>
            <ArticleImage src={SkipConnectionDiagram2} />
            <p>
              In their experiments, He et al. find that the latter method
              performs slightly better than the former, but point out that the
              former is more economical, because it does not add additional
              parameters to the network. They also point out that a square
              projection matrix could be used to transform{" "}
              <Eq text="$$\textbf{x}$$" /> even when the dimensionality of{" "}
              <Eq text="$$\textbf{x}$$" /> and{" "}
              <Eq text="$$\mathcal{F}(\textbf{x})$$" /> is the same, but they
              find that the performance increase is marginal, so this is
              unnecessary.
            </p>
            <p>
              One final note: the projection in the second method could
              potentially take different forms, and in the ResNet developed by
              He et al., they use <Eq text="$$1 \times 1$$" /> convolutions as
              the linear transformation to increase the dimensionality. In
              particular, their convolutional neural network progressively
              doubles the number of filters while halving the size of each
              feature map. The <Eq text="$$1 \times 1$$" /> convolutions used as
              the linear projections in the skip connections use an appropriate
              number of filters to match the number of filters in the output,
              along with a stride of two to reduce the size of the feature map
              to that of the output. In conjunction with the identity mappings
              in the instances where the dimension does not change, these skip
              connections allow their ResNet to attain excellent performance at
              very little additional cost in learnable parameters.
            </p>
            <ArticleHeader sectionHeader={sectionHeaders[3]} />
            <p>
              Now that we've built up some intuition about why skip connections
              can improve the performance of deep neural networks, let's
              convince ourselves by running a quick experiment. The code I wrote
              for this exercise is built from the{" "}
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

from matplotlib import pyplot as plt`}
            </Prism>
            <p>
              Next, we'll define a "block" of the neural network, which can be
              thought of as set of hidden layers that will repeat itself in the
              full network. To this end, we'll define a class that inherits from{" "}
              <Code>nn.Module</Code> consisting of two linear layers with ReLU
              activation. We want to be able to reuse this block for both the
              non-residual and residual neural networks, its constructor will
              accept a boolean parameter "residual," which specifies whether or
              not to add the original input to the output vector (i.e. whether
              or not to include a skip connection).
            </p>
            <Prism language="python">
              {`class Block(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual
        self.linear1 = nn.Linear(800, 800)
        self.linear2 = nn.Linear(800, 800)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.linear1(x))
        if self.residual:
            out = self.relu(self.linear2(out) + x)
        else:
            out = self.relu(self.linear2(out))
        return out`}
            </Prism>
            <p>
              Notice that if <Code>residual</Code> is set to True, the module
              will add the input to the layer <Code>x</Code> to the output after
              two layers, but before the second ReLU activation function. Next,
              we'll define the full neural network as another class inheriting
              from <Code>nn.Module</Code>; the constructor should accept both a{" "}
              <Code>residual</Code> parameter and a <Code>num_blocks</Code>{" "}
              parameter to specify how many of the two-layer blocks we defined
              to include in the network. Together, these parameters will allow
              us to compare shallower and deeper neural networks, both with and
              without skip connections. For this experiment, we'll work with the
              MNIST dataset, so we first need to flatten the 28x28 input images
              into one 784-dimensional vector. Then, the input layer will expand
              the flattened input into an 800-dimensional vector. We also define
              an <Code>nn.ModuleList</Code> with as many blocks as specified,
              and an 10-dimensional output layer, corresponding with the number
              of classes of digits in the dataset. Here's the code for the{" "}
              <Code>NeuralNetwork</Code> module:
              <Prism language="python">
                {`class NeuralNetwork(nn.Module):
    def __init__(self, residual, num_blocks):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(28*28, 800)
        self.hidden_layers = nn.ModuleList([
            Block(residual) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(800, 10)

    def forward(self, x):
        out = self.flatten(x)
        out = self.input_layer(out)
        for layer in self.hidden_layers:
            out = layer(out)
        logits = self.output_layer(out)
        return logits`}
              </Prism>
            </p>
            <p>
              Next, we'll define two straightforward functions to train the
              neural network and evaluate its performance. These functions are
              taken almost directly from the PyTorch Quickstart Guide.
            </p>
            <Prism language="python">
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
              Lastly, we download the dataset, initialize the models, run the
              training, and plot the results. Again, much of this code is from
              the PyTorch Quickstart Guide, so nothing here should be too
              unfamiliar. I chose to test depths of 5 and 8 for each variant of
              the neural network, but feel free to experiment with different
              values. I used the Adam optimizer instead of manually finding an
              appopriate learning rate scheudle to keep things simple.
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
    
    models = {
        "non-residual-5-blocks": NeuralNetwork(residual=False, num_blocks=5),
        "residual-5-blocks": NeuralNetwork(residual=True, num_blocks=5),
        "non-residual-8-blocks": NeuralNetwork(residual=False, num_blocks=8),
        "residual-8-blocks": NeuralNetwork(residual=True, num_blocks=8),        
    }

    for (model_name, model) in models.items():
        print(f"Evaluating {model_name} model.")

        model.to(device)
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        test_accs = []

        epochs = 25
        for t in range(epochs):
            print(f"Epoch {t+1}\\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            acc = test(test_dataloader, model, loss_fn)
            test_accs.append(acc)
        print("Done!")

        plt.plot(test_accs, label=model_name)
    
    plt.xlabel("Training Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison for MLPs With and Without Skip Connections")
    plt.legend()
    plt.savefig(f"test_acc_plot")`}
            </Prism>
            <p>Here are the results of the training:</p>
            <ArticleImage src={skip_connection_experiment_graph} width="60%" />
            <p>
              The first observation we might make is that, among the two
              non-residual neural networks, the deeper, 8-block network
              performed far worse than its shallower counterpart. However, among
              the networks which included skip connections, the deeper version
              and the shallower version performed about the same. If we trained
              on a more challenging dataset, we would likely see the deeper
              neural network perform better than the shallower one, as He et al.
              found in their paper. Finally, we can clearly see that among
              networks of the same depth, the residual version outperformed the
              non-residual verison. Skip connections might be a simple idea, but
              their incoporation can make a surprisingly big difference!
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default SkipConnectionPage;

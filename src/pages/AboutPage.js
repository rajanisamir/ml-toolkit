const AboutPage = () => {
  return (
    <div
      style={{
        margin: "auto",
        maxWidth: "800px",
      }}
    >
      <h1>About This Project</h1>
      <p>
        As a research field, machine learning is somewhat unique, in that the
        most recent techniques are in many ways essential for any practicing
        engineer to understand. While the volatility of cutting-edge research
        might make it impractical for engineers in other fields to keep up to
        date on the most recent technologies (e.g. a web developer doesn't
        necessarily need to know anything about decentralized web technologies
        or the latest JavaScript framework), it can be critical for a machine
        learning engineer to understand even those methods which have not yet
        reached maturity, because the practicality of their application might
        offset the volatile nature of research.
      </p>
      <p>
        As an example, the transformer neural network architecture
        revolutionized machine learning when the seminal paper{" "}
        <i>Attention is All You Need</i> was published by Vaswani et al. in
        2017. While the transformer's properties are still being studied, with
        improvements constantly being found, the model has also been so dominant
        that it has rendered many previous sequence-to-sequence neural network
        architectures obsolete for certain tasks. Thus, it is the responsibility
        of the engineer to understand the inner workings of the model, despite
        its recency.
      </p>
      <p>
        The intent of this blog is to bridge the gap between the general machine
        learning principles taught in, say, an introductory machine learning
        course, and cutting-edge techniques from recent publications. The
        utilization of these techniques by an engineering both:
      </p>
      <ol>
        <li>
          Confers benefits to the machine learning research community, as
          applying such techniques to disparate fields and data forms expands
          our understanding of their usefulness and breadth of application, and
        </li>
        <li>
          Can help a machine learning engineer tackle problems for which solid
          approaches might not have existed prior to some recent research
          novelty.
        </li>
      </ol>
      <p>
        My other justification for creating this set of tutorials is because of
        a dearth of high-quality educational materials on these more recent
        machine learning developments. To gain a modest amount of intuition for
        a concept, I would often have to jump between reading a paper (where the
        language used might assume familiarity with some more abstruse
        concepts), patching together bits and pieces from different blog posts,
        and scrubbing through video explanations. While struggling through
        understanding material can be a valuable educational tool, struggling to
        find quality source material is usually a waste of time.
      </p>
      <p>
        So far, I've written several articles on topics like the Transformer
        architecture, dropout regularization, and Monte Carlo tree search.
        Future plans include articles on machine learning fundamentals, a series
        on convolutional neural networks, and more!
      </p>
    </div>
  );
};

export default AboutPage;

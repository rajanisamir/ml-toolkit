import React from "react";
import HeaderAction from "./HeaderAction";

const NavBar = () => {
  return (
    <HeaderAction
      links={[
        {
          label: "Model Architectures",
          links: [
            { link: "/transformer", label: "Transformer" },
            {
              link: "/coming-soon",
              label: "Convolutional Neural Network (CNN)",
            },
            { link: "/coming-soon", label: "Linear Classifier" },
            { link: "/coming-soon", label: "Autoencoder" },
          ],
        },
        {
          label: "Machine Learning Concepts",
          links: [
            {
              link: "/coming-soon",
              label: "Layer Norm, Batch Norm, and All That",
            },
            {
              link: "/coming-soon",
              label: "Unsupervised, Supervised, and Self-Supervised Learning",
            },
            {
              link: "/coming-soon",
              label: "Self-Supervised Learning, In Detail",
            },
            { link: "/coming-soon", label: "Backpropagation" },
          ],
        },
        {
          label: "About",
          link: "/about"
        }
      ]}
    />
  );
};

export default NavBar;

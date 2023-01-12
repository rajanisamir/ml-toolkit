import React from "react";
import HeaderAction from "./HeaderAction";

const NavBar = () => {
  return (
    <HeaderAction
      links={[
        {
          label: "The Transformer",
          links: [
            { link: "/transformer1", label: "Part 1: Paying Attention" },
            {
              link: "/transformer2",
              label: "Part 2: Multi-Head Attention & the Encoder",
            },
            { link: "/transformer3", label: "Part 3: The Decoder" },
          ],
        },
        {
          label: "Convolutional Neural Networks",
          links: [
            {
              link: "/coming-soon",
              label: "Part 1: Motivation and Image Classificiation",
            },
            {
              link: "/coming-soon",
              label: "Part 2: Convolutions and Max Pooling",
            },
            {
              link: "/coming-soon",
              label: "Part 3: Improving Upon an MLP for Digit Recognition",
            },
          ],
        },
        {
          label: "About",
          link: "/about",
        },
      ]}
    />
  );
};

export default NavBar;

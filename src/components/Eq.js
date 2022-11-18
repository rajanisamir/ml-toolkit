import React from "react";

var Latex = require("react-latex");

const Eq = ({ text, displayMode = false }) => {
  return <Latex displayMode={displayMode}>{text}</Latex>;
};

export default Eq;

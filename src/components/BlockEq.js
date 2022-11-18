import React from "react";

import { ScrollArea } from "@mantine/core";

import CustomCopyButton from "./CustomCopyButton";

var Latex = require("react-latex");

const BlockEq = ({ text, displayMode = false }) => {
  return (
    <ScrollArea style={{ width: "100%", margin: "auto" }}>
      <div
        style={{
          // backgroundColor: "#FAFAFA",
          padding: "1rem",
          position: "relative",
        }}
      >
        <CustomCopyButton copyValue={text} copyText="Copy LaTeX" />
        <Latex displayMode={displayMode}>{text}</Latex>
      </div>
    </ScrollArea>
  );
};

export default BlockEq;

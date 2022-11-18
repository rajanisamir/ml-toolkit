import "beautiful-react-diagrams/styles.css";
import { getWhitespaceList } from "../utils/DiagramUtils";
import { ScrollArea } from "@mantine/core";

import Diagram, { useSchema, createSchema } from "beautiful-react-diagrams";
import Eq from "../components/Eq";

const words = ["I", "Made", "Up", "A", "Story", ".", "<EOS>"];
const whitespace = getWhitespaceList(words);

const layer0node = {
  id: "query",
  disableDrag: true,
  content: "Query (Made)",
  coordinates: [0, 4 * 80],
  outputs: [{ id: "query-out", alignment: "right", canLink: () => false }],
};

const layer1nodes = words.map((word, i) => {
  return {
    id: `key-${i + 1}`,
    disableDrag: true,
    content: `Key (${word})${whitespace[i]}`,
    coordinates: [300, (i + 1) * 80],
    inputs: [
      { id: `key-${i + 1}-in`, alignment: "left", canLink: () => false },
    ],
    outputs: [
      { id: `key-${i + 1}-out`, alignment: "right", canLink: () => false },
    ],
  };
});

const layer2nodes = words.map((word, i) => {
  return {
    id: `attn-${i + 1}`,
    disableDrag: true,
    content: `Attn. (${word})${whitespace[i]}`,
    coordinates: [500, (i + 1) * 80],
    inputs: [
      { id: `attn-${i + 1}-in`, alignment: "left", canLink: () => false },
    ],
    outputs: [
      { id: `attn-${i + 1}-out`, alignment: "right", canLink: () => false },
    ],
  };
});

const layer3nodes = words.map((word, i) => {
  return {
    id: `value-${i + 1}`,
    disableDrag: true,
    content: `Value (${word})${whitespace[i]}`,
    coordinates: [700, (i + 1) * 80],
    inputs: [
      { id: `value-${i + 1}-in`, alignment: "left", canLink: () => false },
    ],
    outputs: [
      { id: `value-${i + 1}-out`, alignment: "right", canLink: () => false },
    ],
  };
});

const layer4node = {
  id: "embedding",
  disableDrag: true,
  content: "Context-Aware Embedding (Made)",
  coordinates: [1000, 4 * 80],
  inputs: [{ id: "embedding-in", alignment: "left", canLink: () => false }],
};

const layer01links = [1, 2, 3, 4, 5, 6, 7].map((num) => {
  return {
    input: "query-out",
    output: `key-${num}-in`,
    label: <Eq text="$\cdot$" />,
    readonly: true,
  };
});

const layer12links = [1, 2, 3, 4, 5, 6, 7].map((num) => {
  return {
    input: `key-${num}-out`,
    output: `attn-${num}-in`,
    label: "=",
    readonly: true,
  };
});

const layer23links = [1, 2, 3, 4, 5, 6, 7].map((num) => {
  return {
    input: `attn-${num}-out`,
    output: `value-${num}-in`,
    label: <Eq text="$\times$" />,
    readonly: true,
  };
});

const layer34links = [1, 2, 3, 4, 5, 6, 7].map((num) => {
  return {
    input: `value-${num}-out`,
    output: `embedding-in`,
    label: "+",
    readonly: true,
  };
});

const initialSchema = createSchema({
  nodes: [
    layer0node,
    ...layer1nodes,
    ...layer2nodes,
    ...layer3nodes,
    layer4node,
  ],
  links: [...layer01links, ...layer12links, ...layer23links, ...layer34links],
});

const AttentionDiagram = () => {
  // create diagrams schema
  const [schema, { onChange }] = useSchema(initialSchema);

  return (
    <ScrollArea style={{ width: "100%", height: "700px", margin: "auto" }}>
      <div style={{ width: "1200px", height: "700px" }}>
        <Diagram schema={schema} onChange={onChange} />
      </div>
    </ScrollArea>
  );
};

<AttentionDiagram />;

export default AttentionDiagram;

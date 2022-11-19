import "beautiful-react-diagrams/styles.css";

import { ScrollArea } from "@mantine/core";

import Diagram, { useSchema, createSchema } from "beautiful-react-diagrams";

const initialSchema = createSchema({
  nodes: [
    {
      id: "node-1",
      disableDrag: true,
      content: 'Word (e.g. "Make")',
      coordinates: [25, 150],
      outputs: [{ id: "port-1", alignment: "right", canLink: () => false }],
    },
    {
      id: "node-2",
      disableDrag: true,
      content: "Word Embedding",
      coordinates: [250, 150],
      inputs: [{ id: "port-2a", alignment: "left", canLink: () => false }],
      outputs: [{ id: "port-2b", alignment: "right", canLink: () => false }],
    },
    {
      id: "node-3",
      disableDrag: true,
      content: "Query",
      coordinates: [650, 50],
      inputs: [{ id: "port-3", alignment: "left" }],
    },
    {
      id: "node-4",
      disableDrag: true,
      content: "Key",
      coordinates: [650, 150],
      inputs: [{ id: "port-4", alignment: "left" }],
    },
    {
      id: "node-5",
      disableDrag: true,
      content: "Value",
      coordinates: [650, 250],
      inputs: [{ id: "port-5", alignment: "left" }],
    },
  ],
  links: [
    {
      input: "port-1",
      output: "port-2a",
      readonly: true,
    },
    {
      input: "port-2b",
      output: "port-3",
      label: "Query Transformation",
      readonly: true,
    },
    {
      input: "port-2b",
      output: "port-4",
      label: "Key Transformation",
      readonly: true,
    },
    {
      input: "port-2b",
      output: "port-5",
      label: "Value Transformation",
      readonly: true,
    },
  ],
});

const QKVDiagram = () => {
  // create diagrams schema
  const [schema, { onChange }] = useSchema(initialSchema);

  return (
    <ScrollArea style={{ width: "100%", margin: "auto" }}>
      <div style={{ width: "50rem", height: "22.5rem" }}>
        <Diagram schema={schema} onChange={onChange} />
      </div>
    </ScrollArea>
  );
};

<QKVDiagram />;

export default QKVDiagram;

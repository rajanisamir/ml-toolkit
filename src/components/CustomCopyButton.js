import { CopyButton, ActionIcon, Tooltip } from "@mantine/core";
import { IconCopy, IconCheck } from "@tabler/icons";

const CustomCopyButton = ({ copyValue, copyText = "Copy" }) => {
  return (
    <CopyButton value={copyValue} timeout={2000}>
      {({ copied, copy }) => (
        <Tooltip
          label={copied ? "Copied!" : copyText}
          withArrow
          position="right"
        >
          <ActionIcon
            color={copied ? "teal" : "gray"}
            onClick={copy}
            style={{ position: "absolute", top: "1rem", right: "1rem" }}
          >
            {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
          </ActionIcon>
        </Tooltip>
      )}
    </CopyButton>
  );
};

export default CustomCopyButton;

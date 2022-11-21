import { Text } from "@mantine/core";

const ArticleTitle = ({ name }) => {
  return (
    <Text variant="gradient" weight={700} style={{ fontSize: 45 }}>
      {name}
    </Text>
  );
};

export default ArticleTitle;

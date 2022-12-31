import { Text } from "@mantine/core";

const ArticleTitle = ({ name }) => {
  return (
    <Text variant="gradient" weight={700} style={{ fontSize: 55 }}>
      {name}
    </Text>
  );
};

export default ArticleTitle;

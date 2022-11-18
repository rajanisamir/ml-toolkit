import { Text } from "@mantine/core";

const ArticleTitle = ({ title }) => {
  return (
    <Text variant="gradient" weight={700} style={{ fontSize: 45 }}>
      {title}
    </Text>
  );
};

export default ArticleTitle;

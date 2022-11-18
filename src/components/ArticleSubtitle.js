import { Text } from "@mantine/core";

const ArticleSubtitle = ({ title }) => {
  return (
    <Text
      variant="gradient"
      gradient={{ from: "gray.6", to: "gray.8", deg: 45 }}
      weight={700}
      style={{ fontSize: 20 }}
    >
      {title}
    </Text>
  );
};

export default ArticleSubtitle;

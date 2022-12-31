import { Text } from "@mantine/core";

const ArticleSubtitle = ({ name }) => {
  return (
    <Text
      variant="gradient"
      gradient={{ from: "gray.6", to: "gray.8", deg: 45 }}
      weight={700}
      style={{ fontSize: 28, marginBottom: "2rem" }}
    >
      {name}
    </Text>
  );
};

export default ArticleSubtitle;

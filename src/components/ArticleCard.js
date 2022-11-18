import { Card, Image, Text, Badge, Button, Group } from "@mantine/core";
import { useNavigate } from "react-router-dom";

const ArticleCard = ({
  name,
  description,
  inDevelopment = false,
  comingSoon = false,
  img,
  pagePath,
}) => {
  const navigate = useNavigate();

  return (
    <Card shadow="sm" p="lg" radius="md" withBorder>
      <Card.Section>
        <Image src={img} height={160} alt={`${name} Thumbnail`} />
      </Card.Section>

      <Group position="apart" mt="md" mb="xs">
        <Text weight={500}>{name}</Text>
        {inDevelopment ? (
          <Badge color="yellow" variant="light">
            In Development
          </Badge>
        ) : comingSoon ? (
          <Badge color="pink" variant="light">
            Coming Soon
          </Badge>
        ) : (
          ""
        )}
      </Group>

      <Text size="sm" color="dimmed">
        {description}
      </Text>

      <Button
        variant="light"
        color={comingSoon ? "gray" : "blue"}
        style={{ cursor: comingSoon ? "default" : "pointer" }}
        fullWidth
        mt="md"
        radius="md"
        onClick={comingSoon ? null : () => navigate(pagePath)}
      >
        {comingSoon
          ? "Coming Soon"
          : inDevelopment
          ? "Preview Article"
          : "Start Learning"}
      </Button>
    </Card>
  );
};

export default ArticleCard;

import { Group, Avatar, Text } from "@mantine/core";
import profile_picture from "../images/samir_profile_picture.jpg";

const ArticleAuthor = () => {
  return (
    <Group>
      <Avatar src={profile_picture} size={40} />
      <div>
        <Text>Samir Rajani</Text>
        <Text size="xs" color="dimmed">
          Author
        </Text>
      </div>
    </Group>
  );
};

export default ArticleAuthor;

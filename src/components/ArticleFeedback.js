import { Textarea, Button } from "@mantine/core";

const ArticleFeedback = () => {
  return (
    <>
      <Textarea
        placeholder="Loved what you read? Confused by something? Let me know!"
        label="Comments"
        withAsterisk
      />
      <Button mt="md" onClick={() => console.log("Unimplemented")}>
        Submit
      </Button>
    </>
  );
};

export default ArticleFeedback;

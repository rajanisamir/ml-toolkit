import { Image } from "@mantine/core";

const ArticleImage = ({ src, width = "100%", caption, float = "none" }) => {
  return (
    <Image
      src={src}
      style={{
        margin: `4rem ${float === "none" ? "auto" : "4rem"}`,
        width: width,
        float: float,
      }}
      caption={caption}
    />
  );
};

export default ArticleImage;

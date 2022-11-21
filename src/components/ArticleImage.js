import { Image } from "@mantine/core";

const ArticleImage = ({ src, width="100%", caption }) => {
  return (
    <Image src={src} style={{margin: "4rem auto", width: width}} caption={caption} />
  );
};

export default ArticleImage;

const ArticleHeader = ({ sectionHeader }) => {
  return <h3 id={sectionHeader.id}>{sectionHeader.name}</h3>;
};

export default ArticleHeader;

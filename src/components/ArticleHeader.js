const ArticleHeader = ({ sectionHeader }) => {
  return <h3 id={sectionHeader.id} style={{marginTop: "4rem"}}>{sectionHeader.name}</h3>;
};

export default ArticleHeader;

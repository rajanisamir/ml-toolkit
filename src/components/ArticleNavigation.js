import { useState, useEffect } from "react";
import { Box, NavLink } from "@mantine/core";
// import { Link } from "react-scroll";
import { useWindowScroll } from "@mantine/hooks";

const ArticleNavigation = ({ sectionHeaders, getHeaders }) => {
  const [active, setActive] = useState(0);
  const [scroll, scrollTo] = useWindowScroll();

  let headerScrollPostions;

  useEffect(() => {
    headerScrollPostions = [...getHeaders()].map((header) => header.offsetTop);
    for (let i = 0; i < headerScrollPostions.length; i++) {
      if (
        scroll.y > headerScrollPostions[i] &&
        (i + 1 === headerScrollPostions.length ||
          scroll.y < headerScrollPostions[i + 1])
      ) {
        setActive(i);
        break;
      }
    }
  }, [scroll, getHeaders]);

  return (
    <Box className="article-navigation hidden-mobile">
      <h4>Article Contents</h4>
      {sectionHeaders.map((sectionHeader, index) => (
        // <Link key={sectionHeader.id} to={sectionHeader.id} offset={-80} smooth>
        <NavLink
          label={sectionHeader.name}
          onClick={() => {
            scrollTo({ y: headerScrollPostions[index] + 50 });
          }}
          active={active === index}
        />
        // </Link>
      ))}
    </Box>
  );
};

export default ArticleNavigation;

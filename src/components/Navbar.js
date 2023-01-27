import React from "react";
import HeaderAction from "./HeaderAction";

const NavBar = () => {
  return (
    <HeaderAction
      links={[
        {
          label: "About This Project",
          link: "/about",
        },
      ]}
    />
  );
};

export default NavBar;

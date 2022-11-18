import React from "react";
import { useState, useEffect } from "react";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";

const ThemeButton = ({ defaultState = "light" }) => {
  const [theme, setTheme] = useState(defaultState);

  useEffect(() => {
    document.body.classList.add(defaultState);
  });

  const applyLightTheme = () => {
    document.body.classList.remove("dark");
    document.body.classList.add("light");
    setTheme("light");
  };

  const applyDarkTheme = () => {
    document.body.classList.remove("light");
    document.body.classList.add("dark");
    setTheme("dark");
  };

  return theme === "light" ? (
    <DarkModeIcon onClick={applyDarkTheme} className="theme-button" />
  ) : (
    <LightModeIcon onClick={applyLightTheme} className="theme-button" />
  );
};

export default ThemeButton;

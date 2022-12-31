
import { useState } from 'react';

import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import { AppShell, MantineProvider, ColorSchemeProvider } from "@mantine/core";

import ScrollToTop from "./components/ScrollToTop";
import Navbar from "./components/Navbar";

import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import TransformerPage1 from "./pages/TransformerPage1";
import TransformerPage2 from "./pages/TransformerPage2";
import ComingSoonPage from "./pages/ComingSoonPage";

function App() {
  const [colorScheme, setColorScheme] = useState('light');
  const toggleColorScheme = (value) =>
    setColorScheme(value || (colorScheme === 'dark' ? 'light' : 'dark'));

  return (
    <ColorSchemeProvider colorScheme={colorScheme} toggleColorScheme={toggleColorScheme}>
      <MantineProvider
        withGlobalStyles
        withNormalizeCSS
        theme={{
          primaryColor: "indigo",
          fontSizes: {
            xs: 12,
            sm: 14,
            md: 18,
            lg: 24,
            xl: 28,
          },
          defaultGradient: {
            from: "cyan",
            to: "indigo",
            deg: 45,
          },
          colorScheme
        }}
      >
        <Router basename={process.env.PUBLIC_URL}>
          <div>
            <AppShell
              padding="md"
              header={<Navbar />}
              styles={(theme) => ({
                main: {
                  backgroundColor:
                    theme.colorScheme === "dark"
                      ? theme.colors.dark[8]
                      : theme.colors.gray[0],
                },
              })}
            >
              <div className="content">
                <ScrollToTop>
                  <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/about" element={<AboutPage />} />
                    <Route path="/transformer1" element={<TransformerPage1 />} />
                    <Route path="/transformer2" element={<TransformerPage2 />} />
                    <Route path="/coming-soon" element={<ComingSoonPage />} />
                  </Routes>
                </ScrollToTop>
              </div>
            </AppShell>
          </div>
        </Router>
      </MantineProvider>
    </ColorSchemeProvider>
  );
}

export default App;

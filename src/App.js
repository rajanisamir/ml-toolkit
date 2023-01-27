import { useState } from "react";

import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import { AppShell, MantineProvider, ColorSchemeProvider } from "@mantine/core";

import ScrollToTop from "./components/ScrollToTop";
import Navbar from "./components/Navbar";

import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import TransformerPage1 from "./pages/TransformerPage1";
import TransformerPage2 from "./pages/TransformerPage2";
import TransformerPage3 from "./pages/TransformerPage3";
import DropoutPage from "./pages/DropoutPage";
import SkipConnectionPage from "./pages/SkipConnectionPage";
import AutoencoderPage from "./pages/AutoencoderPage";
import TTTIntroPage from "./pages/TTTIntroPage";
import MinimaxPage from "./pages/MinimaxPage";
import ComingSoonPage from "./pages/ComingSoonPage";

function App() {
  const [colorScheme, setColorScheme] = useState("light");
  const toggleColorScheme = (value) =>
    setColorScheme(value || (colorScheme === "dark" ? "light" : "dark"));

  return (
    <ColorSchemeProvider
      colorScheme={colorScheme}
      toggleColorScheme={toggleColorScheme}
    >
      <MantineProvider
        withGlobalStyles
        withNormalizeCSS
        theme={{
          primaryColor: "indigo",
          fontSizes: {
            xs: 12,
            sm: 14,
            md: 17,
            lg: 24,
            xl: 28,
          },
          defaultGradient: {
            from: "cyan",
            to: "indigo",
            deg: 45,
          },
          colorScheme,
        }}
      >
        <Router basename={process.env.PUBLIC_URL}>
          <div>
            <AppShell padding="md" header={<Navbar />}>
              <div className="content">
                <ScrollToTop>
                  <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/about" element={<AboutPage />} />
                    <Route
                      path="/transformer1"
                      element={<TransformerPage1 />}
                    />
                    <Route
                      path="/transformer2"
                      element={<TransformerPage2 />}
                    />
                    <Route
                      path="/transformer3"
                      element={<TransformerPage3 />}
                    />
                    <Route path="/dropout" element={<DropoutPage />} />
                    <Route
                      path="/skip-connections"
                      element={<SkipConnectionPage />}
                    />
                    <Route path="/autoencoders" element={<AutoencoderPage />} />
                    <Route path="/ttt-intro" element={<TTTIntroPage />} />
                    <Route path="/minimax" element={<MinimaxPage />} />
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


import { useState } from 'react';

import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import TransformerPage from "./pages/TransformerPage";
import ComingSoonPage from "./pages/ComingSoonPage";
import ScrollToTop from "./components/ScrollToTop";
import Navbar from "./components/Navbar";
import { AppShell, MantineProvider, ColorSchemeProvider } from "@mantine/core";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";


function App() {
  const [colorScheme, setColorScheme] = useState('dark');
  const toggleColorScheme = (value) =>
    setColorScheme(value || (colorScheme === 'dark' ? 'light' : 'dark'));

  return (
    <ColorSchemeProvider colorScheme={colorScheme} toggleColorScheme={toggleColorScheme}>
      <MantineProvider
        withGlobalStyles
        withNormalizeCSS
        theme={{
          primaryColor: "indigo",
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
                    <Route path="/transformer" element={<TransformerPage />} />
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

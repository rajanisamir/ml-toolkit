import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";
import TransformerPage from "./pages/TransformerPage";
import ComingSoonPage from "./pages/ComingSoonPage";
import ScrollToTop from "./components/ScrollToTop";
import Navbar from "./components/Navbar";
import { AppShell } from "@mantine/core";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

function App() {
  return (
    <Router>
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
  );
}

export default App;

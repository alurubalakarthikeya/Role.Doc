import { BrowserRouter, Routes, Route } from "react-router-dom";
import UploadPage from "./pages/UploadePage";
import ChatPage from "./pages/ChatPage";
import "./styles/Chat.css"; // <-- Optional global styles

function App() {
  return (
    <div className="app-root">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;

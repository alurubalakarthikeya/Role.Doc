// src/pages/DocsPage.jsx
import React from "react";
import "../styles/Docs.css";
import Navbar from "./Navbar";
import docIllustration from "../assets/docs-illustration.svg"; // Add this asset too

export default function DocsPage() {
  return (
    <div className="docs-page">
      <Navbar />
      <div className="hero">
      <section className="docs-hero">
        <div className="docs-text">
          <h1>Documentation.</h1>
          <p>Your go-to guide for working with RoleDoc.</p>
        </div>
        <div className="docs-illustration">
          <img src={docIllustration} alt="Docs Illustration" />
        </div>
      </section>

      <div className="docs-content">
        <section>
          <h2>Getting Started</h2>
          <p>To begin, visit the Upload page and choose a supported document format (PDF, DOCX, TXT). Uploading is fast and secure, and your file stays local unless otherwise specified.</p>
        </section>

        <section>
          <h2>Uploading a File</h2>
          <ul>
            <li>Click the “Upload” button in the navigation bar.</li>
            <li>Drag-and-drop your document or use the file picker.</li>
            <li>Only PDFs, Word docs (.docx), or plain text (.txt) files are allowed.</li>
          </ul>
        </section>

        <section>
          <h2>Chatting with the Document</h2>
          <p>Once uploaded, you’ll be redirected to the chat interface. You can ask natural language questions about the content of your file — like “Summarize this document” or “What are the main points from chapter 2?”</p>
        </section>

        <section>
          <h2>Tips for Best Results</h2>
          <ul>
            <li>Use clean, text-based files (avoid scanned PDFs).</li>
            <li>Ask specific questions for sharper answers.</li>
            <li>PDFs usually yield better formatting in responses.</li>
          </ul>
        </section>

        <section>
          <h2>FAQ</h2>
          <p><strong>Is my document stored?</strong> No, documents are processed locally unless you implement server storage.</p>
          <p><strong>Can I upload multiple files?</strong> Currently, we support one file at a time. Multi-doc chat is in the works.</p>
        </section>
      </div>
    </div>
    </div>
  );
}

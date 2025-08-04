// src/pages/UploadPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Upload.css";
import Navbar from "../pages/Navbar";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const allowedTypes = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
      ];
      if (allowedTypes.includes(selectedFile.type)) {
        setFile(selectedFile);
      } else {
        alert("Only PDF, DOCX, and TXT files are allowed.");
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      // Animate fake progress
      let prog = 0;
      const interval = setInterval(() => {
        prog += 10;
        setProgress(prog);
        if (prog >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            navigate("/chat", {
              state: {
                fileName: file.name,
                fileUrl: URL.createObjectURL(file),
              },
            });
          }, 500);
        }
      }, 100);
    } catch (err) {
      alert("Something went wrong while uploading the file.");
      setUploading(false);
    }
  };

  return (
    <div className="main-content">
      <Navbar hideLinks={["Home", "Upload"]} />
      <div className="upload-box">
        <h1 className="upload-title">Upload a Document</h1>
        <p className="upload-subtitle">PDF, DOCX, or TXT only.</p>

        <label className="upload-label">
          <input
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleFileChange}
            className="hidden-input"
          />
          <p>{file ? file.name : "Click or drag a file here to upload"}</p>
        </label>

        {file && !uploading && (
          <button onClick={handleUpload} className="button-now">
            Upload & Continue
          </button>
        )}

        {uploading && (
          <div className="progress-section">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="progress-text">{progress}% uploaded</p>
          </div>
        )}
      </div>
    </div>
  );
}

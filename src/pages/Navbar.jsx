import React, { useState } from 'react';
import "../styles/Navbar.css";
import Logo from "../assets/logo.png";

export default function Navbar({ hideLinks = [] }) {
  const [menuOpen, setMenuOpen] = useState(false);

  const links = [
    { name: "Home", path: "/" },
    { name: "Upload", path: "/upload" },
    { name: "Docs", path: "/docs" },
    { name: "About", path: "/about" }
  ];

  const filteredLinks = links.filter(link => !hideLinks.includes(link.name));

  return (
    <nav className="navbar">
      <div className="logo">
        <img className="logo-img" src={Logo} alt="RoleDoc Logo" />
      </div>

      <ul className={`nav-links ${menuOpen ? 'open' : ''}`}>
        {filteredLinks.map(link => (
          <li key={link.name}>
            <a href={link.path}>{link.name}</a>
          </li>
        ))}
      </ul>

      <div className="hamburger" onClick={() => setMenuOpen(!menuOpen)}>
        ☰
      </div>
    </nav>
  );
}



import React from 'react';
import '../styles/Preloader.css';
import logo from '../assets/logoo.png'; 

const Preloader = () => {
  return (
    <div id="preloader">
      <div className="ripple-circle">
        <img src={logo} alt="logo" className="ripple-logo" />
      </div>
    </div>
  );
};

export default Preloader;

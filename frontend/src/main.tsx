import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import { MusicPlayerProvider } from './player/MusicPlayerContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <MusicPlayerProvider>
      <App />
    </MusicPlayerProvider>
  </React.StrictMode>,
);

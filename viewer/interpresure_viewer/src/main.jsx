import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import BibleAnalyzer from './viewer.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BibleAnalyzer />
  </StrictMode>,
)

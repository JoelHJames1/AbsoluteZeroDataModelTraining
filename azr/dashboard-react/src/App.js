// Author: Joel Hernandez James  
// Current Date: 2025-05-11  
// Class: App

// Description:  
// Main application component for the AZR dashboard

import React, { useEffect } from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { useThemeContext } from './contexts/ThemeContext';
import { useDataContext } from './contexts/DataContext';
import Header from './components/Header';
import StatusPanel from './components/StatusPanel';
import ChartsPanel from './components/ChartsPanel';
import TasksPanel from './components/TasksPanel';
import BenchmarkPanel from './components/BenchmarkPanel';
import './styles/App.css';

function App() {
  const { darkMode } = useThemeContext();
  const { connectToSocket } = useDataContext();

  // Create MUI theme based on dark mode preference
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#007bff',
      },
      secondary: {
        main: '#6c757d',
      },
      success: {
        main: '#28a745',
      },
      error: {
        main: '#dc3545',
      },
      warning: {
        main: '#ffc107',
      },
      info: {
        main: '#17a2b8',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f7fa',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
      text: {
        primary: darkMode ? '#e0e0e0' : '#333333',
        secondary: darkMode ? '#a0a0a0' : '#6c757d',
      },
    },
    typography: {
      fontFamily: '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      h1: {
        fontSize: '2.2rem',
        fontWeight: 600,
      },
      h2: {
        fontSize: '1.4rem',
        fontWeight: 600,
        marginBottom: '1rem',
      },
      h3: {
        fontSize: '1.2rem',
        fontWeight: 600,
      },
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            boxShadow: darkMode 
              ? '0 4px 6px rgba(0, 0, 0, 0.3)' 
              : '0 4px 6px rgba(0, 0, 0, 0.1)',
            transition: 'all 0.3s ease',
            '&:hover': {
              boxShadow: darkMode 
                ? '0 8px 16px rgba(0, 0, 0, 0.4)' 
                : '0 8px 16px rgba(0, 0, 0, 0.2)',
              transform: 'translateY(-3px)',
            },
          },
        },
      },
    },
  });

  // Connect to Socket.IO server on component mount
  useEffect(() => {
    connectToSocket();
  }, [connectToSocket]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box className="app-container">
        <Header />
        <Box className="main-content">
          <Box className="left-panel">
            <StatusPanel />
          </Box>
          <Box className="right-panel">
            <ChartsPanel />
          </Box>
        </Box>
        <TasksPanel />
        <BenchmarkPanel />
      </Box>
    </ThemeProvider>
  );
}

export default App;

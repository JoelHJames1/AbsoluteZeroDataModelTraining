// Author: Joel Hernandez James  
// Current Date: 2025-05-11  
// Class: Header

// Description:  
// Header component for the AZR dashboard

import React from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  Chip, 
  IconButton, 
  Tooltip,
  useTheme
} from '@mui/material';
import { 
  Brightness4 as DarkModeIcon, 
  Brightness7 as LightModeIcon,
  Code as CodeIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Devices as DevicesIcon
} from '@mui/icons-material';
import { useThemeContext } from '../contexts/ThemeContext';

const Header = () => {
  const theme = useTheme();
  const { darkMode, toggleDarkMode } = useThemeContext();

  return (
    <AppBar 
      position="static" 
      color="transparent" 
      elevation={0}
      sx={{ 
        borderBottom: `1px solid ${theme.palette.divider}`,
        mb: 3,
        backgroundColor: theme.palette.background.paper
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography 
            variant="h1" 
            color="primary" 
            sx={{ 
              fontSize: '2.2rem', 
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <CodeIcon sx={{ mr: 1, fontSize: '2rem' }} />
            Absolute Zero Reasoner
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            icon={<CodeIcon />}
            label="Qwen3-4B"
            color="primary"
            variant="filled"
          />
          <Chip
            icon={<MemoryIcon />}
            label="Self-Play"
            color="success"
            variant="filled"
          />
          <Chip
            icon={<SpeedIcon />}
            label="8-bit"
            color="warning"
            variant="filled"
            sx={{ color: theme.palette.mode === 'dark' ? '#000' : undefined }}
          />
          <Chip
            icon={<DevicesIcon />}
            label="Apple Silicon"
            color="error"
            variant="filled"
          />
          <Tooltip title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}>
            <IconButton onClick={toggleDarkMode} color="inherit">
              {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;

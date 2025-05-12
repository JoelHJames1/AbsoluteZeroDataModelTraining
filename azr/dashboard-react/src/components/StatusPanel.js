// Author: Joel Hernandez James  
// Current Date: 2025-05-11  
// Class: StatusPanel

// Description:  
// Status panel component showing training status and metrics

import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Grid,
  Paper,
  useTheme
} from '@mui/material';
import { useDataContext } from '../contexts/DataContext';

const StatusPanel = () => {
  const theme = useTheme();
  const { trainingData } = useDataContext();
  const [elapsedTime, setElapsedTime] = useState('00:00:00');
  
  // Format elapsed time as HH:MM:SS
  useEffect(() => {
    const startTime = new Date();
    startTime.setSeconds(startTime.getSeconds() - trainingData.currentStep);
    
    const timer = setInterval(() => {
      const now = new Date();
      const elapsed = Math.floor((now - startTime) / 1000);
      const hours = Math.floor(elapsed / 3600);
      const minutes = Math.floor((elapsed % 3600) / 60);
      const seconds = elapsed % 60;
      
      setElapsedTime(
        `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
      );
    }, 1000);
    
    return () => clearInterval(timer);
  }, [trainingData.currentStep]);
  
  // Determine status indicator class
  const getStatusClass = () => {
    switch (trainingData.status) {
      case 'active':
        return 'active';
      case 'paused':
        return 'paused';
      case 'error':
      case 'disconnected':
        return 'error';
      default:
        return '';
    }
  };
  
  // Determine status text
  const getStatusText = () => {
    switch (trainingData.status) {
      case 'active':
        return 'Active';
      case 'paused':
        return 'Paused';
      case 'error':
        return 'Error';
      case 'disconnected':
        return 'Disconnected';
      default:
        return 'Connecting...';
    }
  };
  
  return (
    <>
      <Card>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Training Status
          </Typography>
          
          <Box className={`status-indicator ${getStatusClass()}`} sx={{ mb: 2 }}>
            <Box 
              className="status-dot" 
              sx={{ 
                width: 12, 
                height: 12, 
                borderRadius: '50%', 
                mr: 1,
                bgcolor: trainingData.status === 'active' 
                  ? theme.palette.success.main 
                  : trainingData.status === 'paused'
                    ? theme.palette.warning.main
                    : theme.palette.error.main,
                boxShadow: trainingData.status === 'active' 
                  ? `0 0 0 3px ${theme.palette.success.main}30` 
                  : 'none',
                animation: trainingData.status === 'active' 
                  ? 'pulse 1.5s infinite' 
                  : 'none'
              }} 
            />
            <Typography 
              className="status-text" 
              sx={{ 
                fontWeight: 600,
                color: trainingData.status === 'active' 
                  ? theme.palette.success.main 
                  : trainingData.status === 'paused'
                    ? theme.palette.warning.main
                    : theme.palette.error.main
              }}
            >
              {getStatusText()}
            </Typography>
          </Box>
          
          <Grid container spacing={2} className="status-details">
            <Grid item xs={6}>
              <Box className="detail">
                <Typography variant="body2" color="textSecondary" className="detail-label">
                  Current Step:
                </Typography>
                <Typography variant="body1" fontWeight="bold" className="detail-value">
                  {trainingData.currentStep.toLocaleString()}
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={6}>
              <Box className="detail">
                <Typography variant="body2" color="textSecondary" className="detail-label">
                  Elapsed Time:
                </Typography>
                <Typography variant="body1" fontWeight="bold" className="detail-value">
                  {elapsedTime}
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={6}>
              <Box className="detail">
                <Typography variant="body2" color="textSecondary" className="detail-label">
                  Current Task Type:
                </Typography>
                <Typography variant="body1" fontWeight="bold" className="detail-value">
                  {trainingData.taskType}
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={6}>
              <Box className="detail">
                <Typography variant="body2" color="textSecondary" className="detail-label">
                  Task Difficulty:
                </Typography>
                <Typography variant="body1" fontWeight="bold" className="detail-value">
                  {trainingData.taskDifficulty.toFixed(2)}
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Performance Metrics
          </Typography>
          
          <Grid container spacing={2} className="metrics">
            <Grid item xs={6}>
              <Paper 
                elevation={0} 
                className="metric" 
                sx={{ 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
                  p: 2,
                  borderRadius: 2
                }}
              >
                <Typography 
                  variant="h3" 
                  color="primary" 
                  className="metric-value"
                  sx={{ fontSize: '1.8rem' }}
                >
                  {(trainingData.successRate * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="textSecondary" className="metric-label">
                  Success Rate
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={6}>
              <Paper 
                elevation={0} 
                className="metric" 
                sx={{ 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
                  p: 2,
                  borderRadius: 2
                }}
              >
                <Typography 
                  variant="h3" 
                  color="primary" 
                  className="metric-value"
                  sx={{ fontSize: '1.8rem' }}
                >
                  {trainingData.avgReward.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="textSecondary" className="metric-label">
                  Avg. Reward
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={6}>
              <Paper 
                elevation={0} 
                className="metric" 
                sx={{ 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
                  p: 2,
                  borderRadius: 2
                }}
              >
                <Typography 
                  variant="h3" 
                  color="primary" 
                  className="metric-value"
                  sx={{ fontSize: '1.8rem' }}
                >
                  {trainingData.tasksSolved.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="textSecondary" className="metric-label">
                  Tasks Solved
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={6}>
              <Paper 
                elevation={0} 
                className="metric" 
                sx={{ 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
                  p: 2,
                  borderRadius: 2
                }}
              >
                <Typography 
                  variant="h3" 
                  color="primary" 
                  className="metric-value"
                  sx={{ fontSize: '1.8rem' }}
                >
                  {trainingData.bufferSize.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="textSecondary" className="metric-label">
                  Buffer Size
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </>
  );
};

export default StatusPanel;

// Author: Joel Hernandez James  
// Current Date: 2025-05-11  
// Class: BenchmarkPanel

// Description:  
// Benchmark panel component showing progress towards benchmark targets

import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  LinearProgress,
  useTheme
} from '@mui/material';
import { useDataContext } from '../contexts/DataContext';

const BenchmarkPanel = () => {
  const theme = useTheme();
  const { trainingData, config, milestones } = useDataContext();
  
  // Get progress bar color based on achievement level
  const getProgressColor = (benchmark, value) => {
    const targets = config.benchmarkTargets[benchmark];
    
    if (value >= targets.azrTarget) {
      return theme.palette.mode === 'dark' ? '#9c27b0' : '#9c27b0'; // Purple for target
    } else if (value >= targets.claude2) {
      return theme.palette.success.main; // Green for Claude 2
    } else if (value >= targets.codellama) {
      return theme.palette.info.main; // Blue for CodeLlama
    } else if (value >= targets.gpt35) {
      return theme.palette.primary.main; // Primary for GPT-3.5
    } else {
      return theme.palette.mode === 'dark' ? '#555' : '#aaa'; // Gray for below all
    }
  };
  
  // Render a benchmark milestone
  const renderMilestone = (name, benchmark) => {
    const value = trainingData.benchmarkProgress?.[benchmark] || 0;
    const targets = config.benchmarkTargets[benchmark];
    const achieved = milestones[benchmark];
    
    return (
      <Box className="milestone" sx={{ mb: 4 }}>
        <Box 
          className="milestone-header" 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 1 
          }}
        >
          <Typography 
            variant="subtitle1" 
            fontWeight="bold" 
            className="milestone-name" 
            sx={{ width: 100 }}
          >
            {name}
          </Typography>
          
          <Box 
            className="progress-container" 
            sx={{ 
              flex: 1, 
              position: 'relative',
              height: 24,
              bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
              borderRadius: 2,
              overflow: 'hidden'
            }}
          >
            <LinearProgress
              variant="determinate"
              value={Math.min(100, (value / targets.azrTarget) * 100)}
              sx={{
                height: '100%',
                borderRadius: 2,
                bgcolor: 'transparent',
                '& .MuiLinearProgress-bar': {
                  bgcolor: getProgressColor(benchmark, value),
                  borderRadius: 2,
                  transition: 'transform 0.5s ease, background-color 0.5s ease',
                }
              }}
              className={
                achieved.target ? 'milestone-pulse-target' : 
                achieved.claude2 ? 'milestone-pulse' : ''
              }
            />
            
            <Typography 
              variant="body2" 
              fontWeight="bold"
              sx={{ 
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                color: theme.palette.getContrastText(getProgressColor(benchmark, value)),
                textShadow: '0 0 2px rgba(0,0,0,0.5)',
                zIndex: 1
              }}
            >
              {value.toFixed(1)}%
            </Typography>
          </Box>
        </Box>
        
        <Box 
          className="milestone-targets" 
          sx={{ 
            position: 'relative', 
            height: 30, 
            ml: 12.5 
          }}
        >
          {/* GPT-3.5 Target */}
          <Box 
            className="target" 
            sx={{ 
              position: 'relative', 
              mb: 0.5 
            }}
          >
            <Box 
              className="target-marker" 
              sx={{ 
                position: 'absolute',
                top: 0,
                left: `${(targets.gpt35 / targets.azrTarget) * 100}%`,
                width: 2,
                height: 10,
                bgcolor: theme.palette.primary.main,
                opacity: achieved.gpt35 ? 1 : 0.5
              }}
            />
            <Typography 
              variant="caption" 
              className="target-label" 
              sx={{ 
                position: 'absolute',
                top: 12,
                left: `${(targets.gpt35 / targets.azrTarget) * 100}%`,
                transform: 'translateX(-50%)',
                color: achieved.gpt35 ? theme.palette.primary.main : theme.palette.text.secondary,
                fontWeight: achieved.gpt35 ? 'bold' : 'normal'
              }}
            >
              GPT-3.5 ({targets.gpt35}%)
            </Typography>
          </Box>
          
          {/* CodeLlama Target */}
          <Box 
            className="target" 
            sx={{ 
              position: 'relative', 
              mb: 0.5 
            }}
          >
            <Box 
              className="target-marker" 
              sx={{ 
                position: 'absolute',
                top: 0,
                left: `${(targets.codellama / targets.azrTarget) * 100}%`,
                width: 2,
                height: 10,
                bgcolor: theme.palette.info.main,
                opacity: achieved.codellama ? 1 : 0.5
              }}
            />
            <Typography 
              variant="caption" 
              className="target-label" 
              sx={{ 
                position: 'absolute',
                top: 12,
                left: `${(targets.codellama / targets.azrTarget) * 100}%`,
                transform: 'translateX(-50%)',
                color: achieved.codellama ? theme.palette.info.main : theme.palette.text.secondary,
                fontWeight: achieved.codellama ? 'bold' : 'normal'
              }}
            >
              CodeLlama ({targets.codellama}%)
            </Typography>
          </Box>
          
          {/* Claude 2 Target */}
          <Box 
            className="target" 
            sx={{ 
              position: 'relative', 
              mb: 0.5 
            }}
          >
            <Box 
              className="target-marker" 
              sx={{ 
                position: 'absolute',
                top: 0,
                left: `${(targets.claude2 / targets.azrTarget) * 100}%`,
                width: 2,
                height: 10,
                bgcolor: theme.palette.success.main,
                opacity: achieved.claude2 ? 1 : 0.5
              }}
            />
            <Typography 
              variant="caption" 
              className="target-label" 
              sx={{ 
                position: 'absolute',
                top: 12,
                left: `${(targets.claude2 / targets.azrTarget) * 100}%`,
                transform: 'translateX(-50%)',
                color: achieved.claude2 ? theme.palette.success.main : theme.palette.text.secondary,
                fontWeight: achieved.claude2 ? 'bold' : 'normal'
              }}
            >
              Claude 2 ({targets.claude2}%)
            </Typography>
          </Box>
          
          {/* AZR Target */}
          <Box 
            className="target" 
            sx={{ 
              position: 'relative' 
            }}
          >
            <Box 
              className="target-marker" 
              sx={{ 
                position: 'absolute',
                top: 0,
                left: '100%',
                width: 2,
                height: 10,
                bgcolor: '#9c27b0',
                opacity: achieved.target ? 1 : 0.5
              }}
            />
            <Typography 
              variant="caption" 
              className="target-label" 
              sx={{ 
                position: 'absolute',
                top: 12,
                left: '100%',
                transform: 'translateX(-50%)',
                color: achieved.target ? '#9c27b0' : theme.palette.text.secondary,
                fontWeight: achieved.target ? 'bold' : 'normal'
              }}
            >
              Target ({targets.azrTarget}%)
            </Typography>
          </Box>
        </Box>
      </Box>
    );
  };
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h2" gutterBottom>
          Benchmark Milestones
        </Typography>
        
        <Box className="milestones">
          {renderMilestone('HumanEval', 'humaneval')}
          {renderMilestone('MBPP', 'mbpp')}
          {renderMilestone('APPS', 'apps')}
        </Box>
      </CardContent>
    </Card>
  );
};

export default BenchmarkPanel;

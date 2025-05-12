// Author: Joel Hernandez James  
// Current Date: 2025-05-11  
// Class: ChartsPanel

// Description:  
// Charts panel component showing training progress and benchmark comparisons

import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  useTheme
} from '@mui/material';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { useDataContext } from '../contexts/DataContext';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ChartsPanel = () => {
  const theme = useTheme();
  const { trainingData, config } = useDataContext();
  
  // Prepare data for training progress chart
  const trainingProgressData = {
    labels: trainingData.trainingData?.steps || [],
    datasets: [
      {
        label: 'Success Rate',
        data: trainingData.trainingData?.successRates || [],
        borderColor: theme.palette.success.main,
        backgroundColor: `${theme.palette.success.main}20`,
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointBackgroundColor: theme.palette.success.main,
        pointBorderColor: theme.palette.background.paper,
        pointRadius: 3,
        pointHoverRadius: 5
      },
      {
        label: 'Average Reward',
        data: trainingData.trainingData?.rewards || [],
        borderColor: theme.palette.primary.main,
        backgroundColor: `${theme.palette.primary.main}20`,
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointBackgroundColor: theme.palette.primary.main,
        pointBorderColor: theme.palette.background.paper,
        pointRadius: 3,
        pointHoverRadius: 5
      }
    ]
  };
  
  // Prepare data for benchmark comparison chart
  const benchmarkData = {
    labels: ['GPT-3.5', 'CodeLlama', 'Claude 2', 'AZR (Current)', 'AZR (Target)'],
    datasets: [
      {
        label: 'HumanEval',
        data: [
          config.benchmarkTargets?.humaneval?.gpt35 || 48.1,
          config.benchmarkTargets?.humaneval?.codellama || 53.7,
          config.benchmarkTargets?.humaneval?.claude2 || 56.0,
          trainingData.benchmarkProgress?.humaneval || 0,
          config.benchmarkTargets?.humaneval?.azrTarget || 67.3
        ],
        backgroundColor: theme.palette.primary.main,
        borderColor: theme.palette.primary.dark,
        borderWidth: 1,
        borderRadius: 4
      },
      {
        label: 'MBPP',
        data: [
          config.benchmarkTargets?.mbpp?.gpt35 || 52.3,
          config.benchmarkTargets?.mbpp?.codellama || 57.2,
          config.benchmarkTargets?.mbpp?.claude2 || 61.5,
          trainingData.benchmarkProgress?.mbpp || 0,
          config.benchmarkTargets?.mbpp?.azrTarget || 72.1
        ],
        backgroundColor: theme.palette.success.main,
        borderColor: theme.palette.success.dark,
        borderWidth: 1,
        borderRadius: 4
      },
      {
        label: 'APPS',
        data: [
          config.benchmarkTargets?.apps?.gpt35 || 27.5,
          config.benchmarkTargets?.apps?.codellama || 31.2,
          config.benchmarkTargets?.apps?.claude2 || 33.8,
          trainingData.benchmarkProgress?.apps || 0,
          config.benchmarkTargets?.apps?.azrTarget || 42.7
        ],
        backgroundColor: theme.palette.error.main,
        borderColor: theme.palette.error.dark,
        borderWidth: 1,
        borderRadius: 4
      }
    ]
  };
  
  // Chart options
  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Training Steps',
          font: {
            weight: 'bold'
          },
          color: theme.palette.text.secondary
        },
        grid: {
          display: true,
          color: theme.palette.divider
        },
        ticks: {
          color: theme.palette.text.secondary
        }
      },
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Value',
          font: {
            weight: 'bold'
          },
          color: theme.palette.text.secondary
        },
        grid: {
          display: true,
          color: theme.palette.divider
        },
        ticks: {
          color: theme.palette.text.secondary
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            size: 12
          },
          color: theme.palette.text.primary
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: theme.palette.mode === 'dark' 
          ? 'rgba(0, 0, 0, 0.8)' 
          : 'rgba(255, 255, 255, 0.8)',
        titleColor: theme.palette.text.primary,
        bodyColor: theme.palette.text.primary,
        borderColor: theme.palette.divider,
        borderWidth: 1,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        padding: 12,
        cornerRadius: 6,
        caretSize: 6
      }
    }
  };
  
  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Models',
          font: {
            weight: 'bold'
          },
          color: theme.palette.text.secondary
        },
        grid: {
          display: false
        },
        ticks: {
          color: theme.palette.text.secondary
        }
      },
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Pass@1 (%)',
          font: {
            weight: 'bold'
          },
          color: theme.palette.text.secondary
        },
        grid: {
          display: true,
          color: theme.palette.divider
        },
        ticks: {
          color: theme.palette.text.secondary
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            size: 12
          },
          color: theme.palette.text.primary
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: theme.palette.mode === 'dark' 
          ? 'rgba(0, 0, 0, 0.8)' 
          : 'rgba(255, 255, 255, 0.8)',
        titleColor: theme.palette.text.primary,
        bodyColor: theme.palette.text.primary,
        borderColor: theme.palette.divider,
        borderWidth: 1,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        padding: 12,
        cornerRadius: 6,
        caretSize: 6
      }
    }
  };
  
  return (
    <>
      <Card>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Training Progress
          </Typography>
          
          <Box className="chart-container" sx={{ height: 300 }}>
            <Line data={trainingProgressData} options={lineChartOptions} />
          </Box>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Benchmark Comparison
          </Typography>
          
          <Box className="chart-container" sx={{ height: 300 }}>
            <Bar data={benchmarkData} options={barChartOptions} />
          </Box>
        </CardContent>
      </Card>
    </>
  );
};

export default ChartsPanel;

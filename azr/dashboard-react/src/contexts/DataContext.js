// Author: Joel Hernandez James  
// Current Date: 2025-05-11  
// Class: DataContext

// Description:  
// Context provider for real-time training data from the AZR system

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import io from 'socket.io-client';
import axios from 'axios';

// API endpoints
const API_URL = 'http://localhost:5000';
const SOCKET_URL = 'http://localhost:5000';

// Create context
const DataContext = createContext();

// Custom hook to use the data context
export const useDataContext = () => useContext(DataContext);

// Data provider component
export const DataProvider = ({ children }) => {
  // Socket.IO instance
  const [socket, setSocket] = useState(null);
  
  // Training data state
  const [trainingData, setTrainingData] = useState({
    status: 'connecting',
    currentStep: 0,
    taskType: 'Deduction',
    taskDifficulty: 0.1,
    successRate: 0,
    avgReward: 0,
    tasksSolved: 0,
    bufferSize: 0,
    recentTasks: [],
    benchmarkProgress: {
      humaneval: 0,
      mbpp: 0,
      apps: 0
    },
    trainingData: {
      steps: [],
      rewards: [],
      successRates: []
    }
  });
  
  // Dashboard configuration
  const [config, setConfig] = useState({
    updateInterval: 1000,
    maxDataPoints: 100,
    benchmarkTargets: {
      humaneval: {
        gpt35: 48.1,
        codellama: 53.7,
        claude2: 56.0,
        azrTarget: 67.3
      },
      mbpp: {
        gpt35: 52.3,
        codellama: 57.2,
        claude2: 61.5,
        azrTarget: 72.1
      },
      apps: {
        gpt35: 27.5,
        codellama: 31.2,
        claude2: 33.8,
        azrTarget: 42.7
      }
    },
    animations: {
      enabled: true,
      duration: 800,
      easing: 'easeOutQuart'
    }
  });
  
  // Milestone achievements
  const [milestones, setMilestones] = useState({
    humaneval: {
      gpt35: false,
      codellama: false,
      claude2: false,
      target: false
    },
    mbpp: {
      gpt35: false,
      codellama: false,
      claude2: false,
      target: false
    },
    apps: {
      gpt35: false,
      codellama: false,
      claude2: false,
      target: false
    }
  });

  // Connect to Socket.IO server
  const connectToSocket = useCallback(() => {
    // Disconnect existing socket if any
    if (socket) {
      socket.disconnect();
    }
    
    // Create new socket connection
    const newSocket = io(SOCKET_URL);
    
    // Socket event handlers
    newSocket.on('connect', () => {
      console.log('Connected to AZR training data server');
      setTrainingData(prev => ({ ...prev, status: 'active' }));
    });
    
    newSocket.on('disconnect', () => {
      console.log('Disconnected from AZR training data server');
      setTrainingData(prev => ({ ...prev, status: 'disconnected' }));
    });
    
    newSocket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      setTrainingData(prev => ({ ...prev, status: 'error' }));
    });
    
    newSocket.on('training_data', (data) => {
      setTrainingData(data);
      checkMilestones(data.benchmarkProgress);
    });
    
    // Save socket instance
    setSocket(newSocket);
    
    // Cleanup on unmount
    return () => {
      newSocket.disconnect();
    };
  }, [socket]);

  // Fetch dashboard configuration
  const fetchConfig = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/api/config`);
      setConfig(response.data);
    } catch (error) {
      console.error('Error fetching dashboard configuration:', error);
    }
  }, []);

  // Check for milestone achievements
  const checkMilestones = useCallback((benchmarkProgress) => {
    setMilestones(prev => {
      const newMilestones = { ...prev };
      
      // Check each benchmark
      Object.entries(benchmarkProgress).forEach(([benchmark, value]) => {
        // Check each target
        if (value >= config.benchmarkTargets[benchmark].gpt35 && !prev[benchmark].gpt35) {
          newMilestones[benchmark].gpt35 = true;
          console.log(`Milestone achieved: ${benchmark} surpassed GPT-3.5 (${config.benchmarkTargets[benchmark].gpt35}%)`);
        }
        
        if (value >= config.benchmarkTargets[benchmark].codellama && !prev[benchmark].codellama) {
          newMilestones[benchmark].codellama = true;
          console.log(`Milestone achieved: ${benchmark} surpassed CodeLlama (${config.benchmarkTargets[benchmark].codellama}%)`);
        }
        
        if (value >= config.benchmarkTargets[benchmark].claude2 && !prev[benchmark].claude2) {
          newMilestones[benchmark].claude2 = true;
          console.log(`Milestone achieved: ${benchmark} surpassed Claude 2 (${config.benchmarkTargets[benchmark].claude2}%)`);
        }
        
        if (value >= config.benchmarkTargets[benchmark].azrTarget && !prev[benchmark].target) {
          newMilestones[benchmark].target = true;
          console.log(`Milestone achieved: ${benchmark} reached target (${config.benchmarkTargets[benchmark].azrTarget}%)`);
        }
      });
      
      return newMilestones;
    });
  }, [config.benchmarkTargets]);

  // Fetch initial data and configuration on mount
  useEffect(() => {
    fetchConfig();
    
    // Fetch initial training data
    const fetchInitialData = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/training-data`);
        setTrainingData(response.data);
        checkMilestones(response.data.benchmarkProgress);
      } catch (error) {
        console.error('Error fetching initial training data:', error);
      }
    };
    
    fetchInitialData();
  }, [fetchConfig, checkMilestones]);

  // Provide data context
  return (
    <DataContext.Provider 
      value={{ 
        trainingData, 
        config, 
        milestones, 
        connectToSocket,
        socket
      }}
    >
      {children}
    </DataContext.Provider>
  );
};

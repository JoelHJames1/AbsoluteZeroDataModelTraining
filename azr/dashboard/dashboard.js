// AZR Dashboard JavaScript - Real-time Training Visualization

// Configuration
const config = {
    updateInterval: 1000, // Update interval in milliseconds (faster updates)
    maxDataPoints: 100,   // Maximum number of data points to display on charts (more data points)
    apiEndpoint: '/api/training-data', // API endpoint for real-time data
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
};

// State
let state = {
    trainingActive: true,
    simulationActive: false,
    currentStep: 0,
    startTime: new Date(),
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
    },
    benchmarkData: {
        labels: ['GPT-3.5', 'CodeLlama', 'Claude 2', 'AZR (Current)', 'AZR (Target)'],
        humaneval: [48.1, 53.7, 56.0, 0, 67.3],
        mbpp: [52.3, 57.2, 61.5, 0, 72.1],
        apps: [27.5, 31.2, 33.8, 0, 42.7]
    },
    milestones: {
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
    }
};

// DOM Elements
const elements = {
    currentStep: document.getElementById('current-step'),
    elapsedTime: document.getElementById('elapsed-time'),
    currentTaskType: document.getElementById('current-task-type'),
    taskDifficulty: document.getElementById('task-difficulty'),
    successRate: document.getElementById('success-rate'),
    avgReward: document.getElementById('avg-reward'),
    tasksSolved: document.getElementById('tasks-solved'),
    bufferSize: document.getElementById('buffer-size'),
    recentTasks: document.getElementById('recent-tasks'),
    trainingProgressChart: document.getElementById('training-progress-chart'),
    benchmarkChart: document.getElementById('benchmark-chart'),
    humanevalProgress: document.getElementById('humaneval-progress'),
    humanevalProgressText: document.getElementById('humaneval-progress-text'),
    mbppProgress: document.getElementById('mbpp-progress'),
    mbppProgressText: document.getElementById('mbpp-progress-text'),
    appsProgress: document.getElementById('apps-progress'),
    appsProgressText: document.getElementById('apps-progress-text')
};

// Charts
let trainingChart;
let benchmarkChart;

// Initialize the dashboard
function initDashboard() {
    // Apply theme based on system preference
    applyTheme();
    
    // Initialize charts with animations
    initCharts();
    
    // Initial dashboard update
    updateDashboard();
    
    // Set up real-time data fetching
    fetchDataAndUpdate();
    setInterval(fetchDataAndUpdate, config.updateInterval);
    
    // Add event listeners for interactive features
    setupEventListeners();
    
    // Add particle background effect
    initParticleBackground();
}

// Apply theme based on system preference
function applyTheme() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.classList.add('dark-theme');
        updateChartTheme('dark');
    }
    
    // Listen for theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (e.matches) {
            document.body.classList.add('dark-theme');
            updateChartTheme('dark');
        } else {
            document.body.classList.remove('dark-theme');
            updateChartTheme('light');
        }
    });
}

// Update chart theme
function updateChartTheme(theme) {
    const textColor = theme === 'dark' ? '#e0e0e0' : '#333333';
    const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = gridColor;
    
    if (trainingChart) {
        trainingChart.options.scales.x.grid.color = gridColor;
        trainingChart.options.scales.y.grid.color = gridColor;
        trainingChart.options.scales.x.ticks.color = textColor;
        trainingChart.options.scales.y.ticks.color = textColor;
        trainingChart.update();
    }
    
    if (benchmarkChart) {
        benchmarkChart.options.scales.x.grid.color = gridColor;
        benchmarkChart.options.scales.y.grid.color = gridColor;
        benchmarkChart.options.scales.x.ticks.color = textColor;
        benchmarkChart.options.scales.y.ticks.color = textColor;
        benchmarkChart.update();
    }
}

// Fetch data from API and update dashboard
function fetchDataAndUpdate() {
    fetch(config.apiEndpoint)
        .then(response => response.json())
        .then(data => {
            // Update state with data from API
            state.currentStep = data.currentStep;
            state.taskType = data.taskType;
            state.taskDifficulty = data.taskDifficulty;
            state.successRate = data.successRate;
            state.avgReward = data.avgReward;
            state.tasksSolved = data.tasksSolved;
            state.bufferSize = data.bufferSize;
            
            // Update benchmark progress
            state.benchmarkProgress = data.benchmarkProgress;
            
            // Update training data for charts
            if (data.trainingData && data.trainingData.steps) {
                state.trainingData = data.trainingData;
            }
            
            // Update recent tasks
            if (data.recentTasks && data.recentTasks.length > 0) {
                state.recentTasks = data.recentTasks;
            }
            
            // Update the dashboard with new data
            updateDashboard();
            
            // Check for milestone achievements
            checkMilestones();
        })
        .catch(error => {
            console.error('Error fetching training data:', error);
            // If API fails, fall back to simulation for demo purposes
            if (!state.simulationActive) {
                console.log('Falling back to simulation mode');
                state.simulationActive = true;
                startSimulation();
            }
        });
}

// Set up event listeners for interactive features
function setupEventListeners() {
    // Add event listener for status card hover effects
    document.querySelectorAll('.status-card, .chart-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.classList.add('card-hover');
        });
        card.addEventListener('mouseleave', () => {
            card.classList.remove('card-hover');
        });
    });
    
    // Add event listener for task item click to show details
    document.querySelector('.task-list').addEventListener('click', (e) => {
        const taskItem = e.target.closest('.task-item');
        if (taskItem) {
            taskItem.classList.toggle('expanded');
        }
    });
}

// Initialize particle background effect
function initParticleBackground() {
    const particleContainer = document.createElement('div');
    particleContainer.className = 'particles-background';
    document.body.prepend(particleContainer);
    
    // Create particles
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random position
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        
        // Random size
        const size = Math.random() * 5 + 2;
        
        // Random opacity
        const opacity = Math.random() * 0.5 + 0.1;
        
        // Random animation duration
        const duration = Math.random() * 20 + 10;
        
        // Set styles
        particle.style.left = `${posX}%`;
        particle.style.top = `${posY}%`;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.opacity = opacity;
        particle.style.animationDuration = `${duration}s`;
        
        particleContainer.appendChild(particle);
    }
}

// Initialize charts with enhanced visuals
function initCharts() {
    // Set global chart defaults for animations
    Chart.defaults.animation = {
        duration: config.animations.duration,
        easing: config.animations.easing
    };
    
    // Training Progress Chart
    const trainingCtx = elements.trainingProgressChart.getContext('2d');
    trainingChart = new Chart(trainingCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Success Rate',
                    data: [],
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.2)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#28a745',
                    pointBorderColor: '#fff',
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Average Reward',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#007bff',
                    pointBorderColor: '#fff',
                    pointRadius: 4,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
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
                        }
                    },
                    grid: {
                        display: true,
                        drawBorder: true,
                        drawOnChartArea: true,
                        drawTicks: true,
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
                        }
                    },
                    grid: {
                        display: true,
                        drawBorder: true,
                        drawOnChartArea: true,
                        drawTicks: true,
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
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
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
        }
    });
    
    // Benchmark Comparison Chart
    const benchmarkCtx = elements.benchmarkChart.getContext('2d');
    benchmarkChart = new Chart(benchmarkCtx, {
        type: 'bar',
        data: {
            labels: state.benchmarkData.labels,
            datasets: [
                {
                    label: 'HumanEval',
                    data: state.benchmarkData.humaneval,
                    backgroundColor: 'rgba(0, 123, 255, 0.8)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 2,
                    borderRadius: 6,
                    hoverBackgroundColor: 'rgba(0, 123, 255, 1)'
                },
                {
                    label: 'MBPP',
                    data: state.benchmarkData.mbpp,
                    backgroundColor: 'rgba(40, 167, 69, 0.8)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 2,
                    borderRadius: 6,
                    hoverBackgroundColor: 'rgba(40, 167, 69, 1)'
                },
                {
                    label: 'APPS',
                    data: state.benchmarkData.apps,
                    backgroundColor: 'rgba(220, 53, 69, 0.8)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 2,
                    borderRadius: 6,
                    hoverBackgroundColor: 'rgba(220, 53, 69, 1)'
                }
            ]
        },
        options: {
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
                        }
                    },
                    grid: {
                        display: false
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
                        }
                    },
                    grid: {
                        display: true,
                        drawBorder: true,
                        drawOnChartArea: true,
                        drawTicks: true,
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
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
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
        }
    });
}

// Update the dashboard with current state and animations
function updateDashboard() {
    // Update status details with smooth transitions
    animateValue(elements.currentStep, state.currentStep);
    elements.elapsedTime.textContent = formatElapsedTime(state.startTime);
    
    if (elements.currentTaskType.textContent !== state.taskType) {
        elements.currentTaskType.classList.add('highlight');
        setTimeout(() => elements.currentTaskType.classList.remove('highlight'), 1000);
        elements.currentTaskType.textContent = state.taskType;
    }
    
    animateValue(elements.taskDifficulty, state.taskDifficulty.toFixed(2));
    
    // Update metrics with animations
    animateValue(elements.successRate, `${(state.successRate * 100).toFixed(1)}%`);
    animateValue(elements.avgReward, state.avgReward.toFixed(2));
    animateValue(elements.tasksSolved, state.tasksSolved);
    animateValue(elements.bufferSize, state.bufferSize);
    
    // Update recent tasks with animations
    updateRecentTasks();
    
    // Update charts with smooth transitions
    updateCharts();
    
    // Update benchmark progress with animations
    updateBenchmarkProgress();
    
    // Update status indicator based on progress
    updateStatusIndicator();
}

// Animate value changes
function animateValue(element, newValue) {
    if (element.textContent !== newValue.toString()) {
        element.classList.add('highlight');
        setTimeout(() => element.classList.remove('highlight'), 1000);
        element.textContent = newValue;
    }
}

// Update status indicator based on progress
function updateStatusIndicator() {
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');
    
    // Remove all status classes
    statusIndicator.classList.remove('active', 'paused', 'completed');
    
    // Determine status based on progress
    if (state.benchmarkProgress.humaneval >= config.benchmarkTargets.humaneval.azrTarget &&
        state.benchmarkProgress.mbpp >= config.benchmarkTargets.mbpp.azrTarget &&
        state.benchmarkProgress.apps >= config.benchmarkTargets.apps.azrTarget) {
        // All targets achieved
        statusIndicator.classList.add('completed');
        statusText.textContent = 'Completed';
    } else if (state.currentStep > 0) {
        // Training in progress
        statusIndicator.classList.add('active');
        statusText.textContent = 'Active';
    } else {
        // Not started or paused
        statusIndicator.classList.add('paused');
        statusText.textContent = 'Paused';
    }
}

// Update recent tasks list with animations
function updateRecentTasks() {
    // Clear placeholder tasks
    if (elements.recentTasks.querySelector('.placeholder')) {
        elements.recentTasks.innerHTML = '';
    }
    
    // Add recent tasks with animations
    state.recentTasks.forEach((task, index) => {
        // Check if task already exists in the DOM
        const existingTask = document.getElementById(`task-${task.id}`);
        if (!existingTask) {
            const taskElement = document.createElement('div');
            taskElement.className = 'task-item new-task';
            taskElement.id = `task-${task.id}`;
            
            taskElement.innerHTML = `
                <div class="task-header">
                    <span class="task-id">Task #${task.id}</span>
                    <span class="task-type">${task.type}</span>
                    <span class="task-difficulty">Difficulty: ${task.difficulty.toFixed(2)}</span>
                    <span class="task-status ${task.solved ? 'success' : 'failure'}">${task.solved ? 'Solved' : 'Failed'}</span>
                </div>
                <div class="task-description">
                    ${task.description}
                </div>
                <div class="task-details">
                    <div class="code-preview">
                        <pre><code>${task.solved ? 
                            `def f(x):\n    # Solution code would appear here\n    pass` : 
                            `# No solution available`}</code></pre>
                    </div>
                    <div class="task-metrics">
                        <div class="task-metric">
                            <span class="task-metric-label">Time:</span>
                            <span class="task-metric-value">${Math.floor(Math.random() * 1000)}ms</span>
                        </div>
                        <div class="task-metric">
                            <span class="task-metric-label">Memory:</span>
                            <span class="task-metric-value">${Math.floor(Math.random() * 100)}KB</span>
                        </div>
                    </div>
                </div>
            `;
            
            // Add to the beginning of the list
            elements.recentTasks.insertBefore(taskElement, elements.recentTasks.firstChild);
            
            // Animate new task
            setTimeout(() => {
                taskElement.classList.remove('new-task');
            }, 10);
            
            // Limit the number of tasks shown
            if (elements.recentTasks.children.length > 10) {
                const lastChild = elements.recentTasks.lastChild;
                lastChild.classList.add('removing');
                setTimeout(() => {
                    if (elements.recentTasks.contains(lastChild)) {
                        elements.recentTasks.removeChild(lastChild);
                    }
                }, 500);
            }
        }
    });
}

// Update charts with current data and smooth animations
function updateCharts() {
    // Update training progress chart with smooth transitions
    trainingChart.data.labels = state.trainingData.steps;
    trainingChart.data.datasets[0].data = state.trainingData.successRates;
    trainingChart.data.datasets[1].data = state.trainingData.rewards;
    
    // Update benchmark chart with smooth transitions
    benchmarkChart.data.datasets[0].data[3] = state.benchmarkProgress.humaneval;
    benchmarkChart.data.datasets[1].data[3] = state.benchmarkProgress.mbpp;
    benchmarkChart.data.datasets[2].data[3] = state.benchmarkProgress.apps;
    
    // Apply animations based on config
    if (config.animations.enabled) {
        trainingChart.update();
        benchmarkChart.update();
    } else {
        trainingChart.update('none');
        benchmarkChart.update('none');
    }
    
    // Add glow effect to charts when reaching milestones
    const chartCards = document.querySelectorAll('.chart-card');
    
    if (state.benchmarkProgress.humaneval >= config.benchmarkTargets.humaneval.gpt35 ||
        state.benchmarkProgress.mbpp >= config.benchmarkTargets.mbpp.gpt35 ||
        state.benchmarkProgress.apps >= config.benchmarkTargets.apps.gpt35) {
        chartCards.forEach(card => card.classList.add('milestone-glow'));
    } else {
        chartCards.forEach(card => card.classList.remove('milestone-glow'));
    }
}

// Update benchmark progress bars with animations
function updateBenchmarkProgress() {
    // Animate progress bars with smooth transitions
    animateProgressBar(elements.humanevalProgress, state.benchmarkProgress.humaneval);
    elements.humanevalProgressText.textContent = `${state.benchmarkProgress.humaneval.toFixed(1)}%`;
    
    animateProgressBar(elements.mbppProgress, state.benchmarkProgress.mbpp);
    elements.mbppProgressText.textContent = `${state.benchmarkProgress.mbpp.toFixed(1)}%`;
    
    animateProgressBar(elements.appsProgress, state.benchmarkProgress.apps);
    elements.appsProgressText.textContent = `${state.benchmarkProgress.apps.toFixed(1)}%`;
    
    // Change color when surpassing benchmarks
    updateProgressBarColors('humaneval', state.benchmarkProgress.humaneval);
    updateProgressBarColors('mbpp', state.benchmarkProgress.mbpp);
    updateProgressBarColors('apps', state.benchmarkProgress.apps);
    
    // Add pulse effect when reaching milestones
    addMilestonePulseEffect('humaneval', state.benchmarkProgress.humaneval);
    addMilestonePulseEffect('mbpp', state.benchmarkProgress.mbpp);
    addMilestonePulseEffect('apps', state.benchmarkProgress.apps);
}

// Animate progress bar width change
function animateProgressBar(element, newValue) {
    element.style.width = `${newValue}%`;
}

// Add pulse effect when reaching milestones
function addMilestonePulseEffect(benchmark, value) {
    const progressBar = document.getElementById(`${benchmark}-progress`);
    const milestones = state.milestones[benchmark];
    
    // Check for GPT-3.5 milestone
    if (value >= config.benchmarkTargets[benchmark].gpt35 && !milestones.gpt35) {
        progressBar.classList.add('milestone-pulse');
        setTimeout(() => progressBar.classList.remove('milestone-pulse'), 2000);
        milestones.gpt35 = true;
        showMilestoneAlert(benchmark, 'GPT-3.5', config.benchmarkTargets[benchmark].gpt35);
    }
    
    // Check for CodeLlama milestone
    if (value >= config.benchmarkTargets[benchmark].codellama && !milestones.codellama) {
        progressBar.classList.add('milestone-pulse');
        setTimeout(() => progressBar.classList.remove('milestone-pulse'), 2000);
        milestones.codellama = true;
        showMilestoneAlert(benchmark, 'CodeLlama', config.benchmarkTargets[benchmark].codellama);
    }
    
    // Check for Claude 2 milestone
    if (value >= config.benchmarkTargets[benchmark].claude2 && !milestones.claude2) {
        progressBar.classList.add('milestone-pulse');
        setTimeout(() => progressBar.classList.remove('milestone-pulse'), 2000);
        milestones.claude2 = true;
        showMilestoneAlert(benchmark, 'Claude 2', config.benchmarkTargets[benchmark].claude2);
    }
    
    // Check for target milestone
    if (value >= config.benchmarkTargets[benchmark].azrTarget && !milestones.target) {
        progressBar.classList.add('milestone-pulse-target');
        setTimeout(() => progressBar.classList.remove('milestone-pulse-target'), 3000);
        milestones.target = true;
        showMilestoneAlert(benchmark, 'Target', config.benchmarkTargets[benchmark].azrTarget, true);
    }
}

// Update progress bar colors based on benchmark targets
function updateProgressBarColors(benchmark, value) {
    const progressBar = document.getElementById(`${benchmark}-progress`);
    
    if (value >= config.benchmarkTargets[benchmark].azrTarget) {
        progressBar.style.backgroundColor = '#9c27b0'; // Purple for exceeding target
    } else if (value >= config.benchmarkTargets[benchmark].claude2) {
        progressBar.style.backgroundColor = '#28a745'; // Green for exceeding Claude 2
    } else if (value >= config.benchmarkTargets[benchmark].codellama) {
        progressBar.style.backgroundColor = '#17a2b8'; // Teal for exceeding CodeLlama
    } else if (value >= config.benchmarkTargets[benchmark].gpt35) {
        progressBar.style.backgroundColor = '#007bff'; // Blue for exceeding GPT-3.5
    } else {
        progressBar.style.backgroundColor = '#6c757d'; // Gray for below all benchmarks
    }
}

// Check for milestone achievements
function checkMilestones() {
    // Check each benchmark for milestones
    checkBenchmarkMilestones('humaneval', state.benchmarkProgress.humaneval);
    checkBenchmarkMilestones('mbpp', state.benchmarkProgress.mbpp);
    checkBenchmarkMilestones('apps', state.benchmarkProgress.apps);
}

// Check benchmark for milestones
function checkBenchmarkMilestones(benchmark, value) {
    const milestones = state.milestones[benchmark];
    
    // Check for GPT-3.5 milestone
    if (value >= config.benchmarkTargets[benchmark].gpt35 && !milestones.gpt35) {
        milestones.gpt35 = true;
        showMilestoneAlert(benchmark, 'GPT-3.5', config.benchmarkTargets[benchmark].gpt35);
    }
    
    // Check for CodeLlama milestone
    if (value >= config.benchmarkTargets[benchmark].codellama && !milestones.codellama) {
        milestones.codellama = true;
        showMilestoneAlert(benchmark, 'CodeLlama', config.benchmarkTargets[benchmark].codellama);
    }
    
    // Check for Claude 2 milestone
    if (value >= config.benchmarkTargets[benchmark].claude2 && !milestones.claude2) {
        milestones.claude2 = true;
        showMilestoneAlert(benchmark, 'Claude 2', config.benchmarkTargets[benchmark].claude2);
    }
    
    // Check for target milestone
    if (value >= config.benchmarkTargets[benchmark].azrTarget && !milestones.target) {
        milestones.target = true;
        showMilestoneAlert(benchmark, 'Target', config.benchmarkTargets[benchmark].azrTarget, true);
    }
}

// Format elapsed time as HH:MM:SS
function formatElapsedTime(startTime) {
    const elapsed = Math.floor((new Date() - startTime) / 1000);
    const hours = Math.floor(elapsed / 3600);
    const minutes = Math.floor((elapsed % 3600) / 60);
    const seconds = elapsed % 60;
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// Simulation for demo purposes
function startSimulation() {
    let simulationStep = 0;
    const maxSteps = 1000;
    const taskTypes = ['Deduction', 'Abduction', 'Induction'];
    
    const simulationInterval = setInterval(() => {
        simulationStep++;
        
        if (simulationStep > maxSteps) {
            clearInterval(simulationInterval);
            return;
        }
        
        // Update state with simulated data
        state.currentStep = simulationStep;
        
        // Gradually increase difficulty
        state.taskDifficulty = Math.min(0.1 + (simulationStep / maxSteps) * 0.9, 0.9);
        
        // Randomly change task type occasionally
        if (simulationStep % 20 === 0) {
            state.taskType = taskTypes[Math.floor(Math.random() * taskTypes.length)];
        }
        
        // Simulate success rate improvement

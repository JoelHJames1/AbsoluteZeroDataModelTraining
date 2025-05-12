#!/bin/bash
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Script: run_dashboard.sh

# Description:  
# Script to run the AZR dashboard with React frontend and Flask API

# Check if the script is run with sudo
if [ "$EUID" -eq 0 ]; then
  echo "Please do not run this script with sudo or as root."
  exit 1
fi

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists node; then
  echo "Node.js is not installed. Please install Node.js to run the dashboard."
  exit 1
fi

if ! command_exists npm; then
  echo "npm is not installed. Please install npm to run the dashboard."
  exit 1
fi

if ! command_exists python3; then
  echo "Python 3 is not installed. Please install Python 3 to run the dashboard."
  exit 1
fi

# Check if Flask and other Python dependencies are installed
python3 -c "import flask, flask_cors, flask_socketio" >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Installing required Python packages..."
  pip install flask flask-cors flask-socketio
fi

# Set up trap to kill background processes on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Change to the script's directory
cd "$(dirname "$0")"

# Check if React dependencies are installed
if [ ! -d "dashboard-react/node_modules" ]; then
  echo "Installing React dependencies..."
  cd dashboard-react
  npm install
  cd ..
fi

# Start the Flask API server
echo "Starting Flask API server..."
python3 api/app.py &
API_PID=$!

# Wait for the API server to start
sleep 2

# Start the React development server
echo "Starting React development server..."
cd dashboard-react
npm start &
REACT_PID=$!

# Wait for both processes
echo "Dashboard is running!"
echo "API server: http://localhost:5000"
echo "React frontend: http://localhost:3000"
echo "Press Ctrl+C to stop both servers."

wait $API_PID $REACT_PID

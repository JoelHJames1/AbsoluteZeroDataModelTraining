#!/usr/bin/env python3
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: AZRFlaskAPI

# Description:  
# Flask API server for the AZR dashboard to provide real-time training data

import os
import sys
import json
import time
import socket
import threading
import logging
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AZR-API")

# Initialize Flask app
app = Flask(__name__, static_folder='../dashboard-react/build')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
server_state = {
    'current_step': 0,
    'task_type': 'Deduction',
    'task_difficulty': 0.1,
    'success_rate': 0,
    'avg_reward': 0,
    'tasks_solved': 0,
    'buffer_size': 0,
    'recent_tasks': [],
    'benchmark_progress': {
        'humaneval': 0,
        'mbpp': 0,
        'apps': 0
    },
    'trainingData': {
        'steps': [],
        'rewards': [],
        'successRates': []
    }
}

# API routes
@app.route('/api/training-data', methods=['GET'])
def get_training_data():
    """API endpoint to get current training data"""
    return jsonify({
        'status': 'active',
        'currentStep': server_state.get('current_step', 0),
        'taskType': server_state.get('task_type', 'Deduction'),
        'taskDifficulty': server_state.get('task_difficulty', 0.1),
        'successRate': server_state.get('success_rate', 0),
        'avgReward': server_state.get('avg_reward', 0),
        'tasksSolved': server_state.get('tasks_solved', 0),
        'bufferSize': server_state.get('buffer_size', 0),
        'recentTasks': server_state.get('recent_tasks', []),
        'benchmarkProgress': server_state.get('benchmark_progress', {
            'humaneval': 0,
            'mbpp': 0,
            'apps': 0
        }),
        'trainingData': server_state.get('trainingData', {
            'steps': [],
            'rewards': [],
            'successRates': []
        })
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """API endpoint to get dashboard configuration"""
    return jsonify({
        'updateInterval': 1000,
        'maxDataPoints': 100,
        'benchmarkTargets': {
            'humaneval': {
                'gpt35': 48.1,
                'codellama': 53.7,
                'claude2': 56.0,
                'azrTarget': 67.3
            },
            'mbpp': {
                'gpt35': 52.3,
                'codellama': 57.2,
                'claude2': 61.5,
                'azrTarget': 72.1
            },
            'apps': {
                'gpt35': 27.5,
                'codellama': 31.2,
                'claude2': 33.8,
                'azrTarget': 42.7
            }
        },
        'animations': {
            'enabled': True,
            'duration': 800,
            'easing': 'easeOutQuart'
        }
    })

# Serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the React app"""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    # Send initial data to the client
    emit('training_data', {
        'status': 'active',
        'currentStep': server_state.get('current_step', 0),
        'taskType': server_state.get('task_type', 'Deduction'),
        'taskDifficulty': server_state.get('task_difficulty', 0.1),
        'successRate': server_state.get('success_rate', 0),
        'avgReward': server_state.get('avg_reward', 0),
        'tasksSolved': server_state.get('tasks_solved', 0),
        'bufferSize': server_state.get('buffer_size', 0),
        'recentTasks': server_state.get('recent_tasks', []),
        'benchmarkProgress': server_state.get('benchmark_progress', {
            'humaneval': 0,
            'mbpp': 0,
            'apps': 0
        }),
        'trainingData': server_state.get('trainingData', {
            'steps': [],
            'rewards': [],
            'successRates': []
        })
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

def update_server_state():
    """Update the server state with real-time training data from the AZR trainer"""
    global server_state
    
    # Connect to the AZR trainer's data server
    host = 'localhost'
    port = 8081
    
    # Initialize connection state
    connected = False
    retry_interval = 5  # seconds
    
    while True:
        try:
            if not connected:
                # Try to connect to the training data server
                logger.info(f"Attempting to connect to AZR training data server at {host}:{port}...")
                
                # Create socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                
                try:
                    # Connect to server
                    sock.connect((host, port))
                    connected = True
                    logger.info("Connected to AZR training data server")
                except (socket.timeout, ConnectionRefusedError) as e:
                    logger.error(f"Could not connect to AZR training data server: {e}")
                    logger.info(f"Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                    continue
            
            # Request training data
            request = "GET /api/training-data HTTP/1.1\r\nHost: localhost\r\n\r\n"
            sock.sendall(request.encode())
            
            # Receive response
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\r\n\r\n" in response:
                    break
            
            # Parse response
            if response:
                try:
                    # Extract JSON data from HTTP response
                    headers, body = response.split(b"\r\n\r\n", 1)
                    data = json.loads(body)
                    
                    # Update server state with training data
                    server_state.update(data)
                    
                    # Print status update
                    logger.info(f"Updated dashboard data: Step {data.get('current_step', 0)}, "
                          f"Success Rate: {data.get('success_rate', 0):.2f}, "
                          f"HumanEval: {data.get('benchmark_progress', {}).get('humaneval', 0):.2f}%")
                    
                    # Emit updated data to all connected clients
                    socketio.emit('training_data', {
                        'status': 'active',
                        'currentStep': server_state['current_step'],
                        'taskType': server_state['task_type'],
                        'taskDifficulty': server_state['task_difficulty'],
                        'successRate': server_state['success_rate'],
                        'avgReward': server_state['avg_reward'],
                        'tasksSolved': server_state['tasks_solved'],
                        'bufferSize': server_state['buffer_size'],
                        'recentTasks': server_state['recent_tasks'],
                        'benchmarkProgress': server_state['benchmark_progress'],
                        'trainingData': server_state['trainingData']
                    })
                    
                except Exception as e:
                    logger.error(f"Error parsing training data: {e}")
                    connected = False
            else:
                logger.warning("No data received from training server")
                connected = False
            
            # Sleep before next update
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error communicating with training server: {e}")
            connected = False
            time.sleep(retry_interval)
            
        finally:
            if not connected and 'sock' in locals():
                sock.close()

def main():
    """Main function to run the API server"""
    global server_state
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Start the update thread
    update_thread = threading.Thread(target=update_server_state, daemon=True)
    update_thread.start()
    
    # Start the Socket.IO server
    port = 5000
    logger.info(f"Starting API server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()

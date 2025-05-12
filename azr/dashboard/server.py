#!/usr/bin/env python3
# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: DashboardServer

# Description:  
# Simple HTTP server to serve the AZR dashboard and provide real-time training data

import http.server
import socketserver
import json
import os
import sys
import threading
import time
import argparse
import webbrowser
from pathlib import Path

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default port
PORT = 8080

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the dashboard server"""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        self.directory = str(Path(__file__).parent.absolute())
        super().__init__(*args, directory=self.directory, **kwargs)
    
    def log_message(self, format, *args):
        """Override to provide cleaner logging"""
        if args[0].startswith('GET /api/'):
            return  # Don't log API requests to reduce noise
        super().log_message(format, *args)
    
    def do_GET(self):
        """Handle GET requests"""
        # API endpoint for training data
        if self.path.startswith('/api/training-data'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get training data from the global state
            data = {
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
                'trainingData': server_state.get('training_data', {
                    'steps': [],
                    'rewards': [],
                    'successRates': []
                })
            }
            
            self.wfile.write(json.dumps(data).encode())
            return
        
        # Serve static files
        return super().do_GET()

def start_server(port=PORT):
    """Start the dashboard server"""
    handler = DashboardHandler
    
    # Try to start the server on the specified port
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Dashboard server started at http://localhost:{port}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Port {port} is already in use. Trying port {port+1}...")
            start_server(port+1)
        else:
            raise

def update_server_state():
    """Update the server state with real-time training data from the AZR trainer"""
    global server_state
    
    # Connect to the AZR trainer's data server
    import socket
    import json
    import time
    
    # Training data server details
    host = 'localhost'
    port = 8081
    
    # Initialize connection state
    connected = False
    retry_interval = 5  # seconds
    
    while True:
        try:
            if not connected:
                # Try to connect to the training data server
                print(f"Attempting to connect to AZR training data server at {host}:{port}...")
                
                # Create socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                
                try:
                    # Connect to server
                    sock.connect((host, port))
                    connected = True
                    print("Connected to AZR training data server")
                except (socket.timeout, ConnectionRefusedError) as e:
                    print(f"Could not connect to AZR training data server: {e}")
                    print(f"Retrying in {retry_interval} seconds...")
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
                    print(f"Updated dashboard data: Step {data.get('currentStep', 0)}, "
                          f"Success Rate: {data.get('successRate', 0):.2f}, "
                          f"HumanEval: {data.get('benchmarkProgress', {}).get('humaneval', 0):.2f}%")
                    
                except Exception as e:
                    print(f"Error parsing training data: {e}")
                    connected = False
            else:
                print("No data received from training server")
                connected = False
            
            # Sleep before next update
            time.sleep(1)
            
        except Exception as e:
            print(f"Error communicating with training server: {e}")
            connected = False
            time.sleep(retry_interval)
            
        finally:
            if not connected and 'sock' in locals():
                sock.close()

def main():
    """Main function to run the dashboard server"""
    global server_state
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AZR Dashboard Server')
    parser.add_argument('--port', type=int, default=PORT, help='Port to run the server on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    # Initialize server state
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
        'training_data': {
            'steps': [],
            'rewards': [],
            'successRates': []
        }
    }
    
    # Start the update thread
    update_thread = threading.Thread(target=update_server_state, daemon=True)
    update_thread.start()
    
    # Open browser if not disabled
    if not args.no_browser:
        webbrowser.open(f'http://localhost:{args.port}')
    
    # Start the server
    start_server(args.port)

if __name__ == "__main__":
    main()

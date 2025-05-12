# Author: Joel Hernandez James  
# Current Date: 2025-05-11  
# Class: Executor

# Description:  
# Safe Python task execution environment for validating code solutions

import multiprocessing
import traceback
import signal
import time

def validate_code(code: str, input_data: str, expected_output: str) -> bool:
    """
    Validates Python code by executing it in a separate process with timeout protection.
    
    Args:
        code: The Python code to validate (should contain a function 'f')
        input_data: The input data to pass to the function
        expected_output: The expected output to compare against
        
    Returns:
        bool: True if the code produces the expected output, False otherwise
    """
    def runner(pipe):
        try:
            local_env = {}
            exec(code, {}, local_env)
            if 'f' not in local_env:
                pipe.send((False, "Function 'f' not defined in the code"))
                return
                
            result = local_env['f'](input_data)
            pipe.send((result == expected_output, result))
        except Exception as e:
            pipe.send((False, f"Exception: {str(e)}\n{traceback.format_exc()}"))
    
    # Create a pipe for communication
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # Create and start the process
    p = multiprocessing.Process(target=runner, args=(child_conn,))
    p.start()
    
    # Wait for the process to complete with a timeout
    p.join(2)  # 2-second timeout
    
    # If the process is still running, terminate it
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    
    # Return the result if available, otherwise False
    if parent_conn.poll():
        result, details = parent_conn.recv()
        return result
    else:
        return False

def execute_with_timeout(code: str, timeout: int = 5) -> tuple:
    """
    Executes arbitrary Python code with timeout protection.
    
    Args:
        code: The Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        tuple: (success, result or error message)
    """
    def runner(pipe):
        try:
            # Create a restricted environment
            local_env = {}
            exec(code, {}, local_env)
            pipe.send((True, "Code executed successfully", local_env.get('result', None)))
        except Exception as e:
            pipe.send((False, f"Exception: {str(e)}\n{traceback.format_exc()}", None))
    
    # Create a pipe for communication
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # Create and start the process
    p = multiprocessing.Process(target=runner, args=(child_conn,))
    p.start()
    
    # Wait for the process to complete with a timeout
    p.join(timeout)
    
    # If the process is still running, terminate it
    if p.is_alive():
        p.terminate()
        p.join()
        return (False, "Execution timed out", None)
    
    # Return the result if available, otherwise an error
    if parent_conn.poll():
        return parent_conn.recv()
    else:
        return (False, "No result received", None)

if __name__ == "__main__":
    # Example usage
    test_code = """
def f(x):
    return x * 2
"""
    print(validate_code(test_code, "5", "10"))
    print(validate_code(test_code, "5", "11"))  # Should fail

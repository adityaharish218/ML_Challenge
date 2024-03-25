#!/usr/bin/python3

"""
Script to distribute the model_trainers. You can generate a config file, start model_trainers 
on all lab machines and kill all running model_testers. The PID_LOAD_TESTER_DIR directory serves
as a lock on the running testers. In case of failure, remaining running
services are identified by their pid file in the pids_model_tester directory.
pid files are named "<host><pid>.pid".
There are three commands:


start:
    Start the model trainers on the different lab machines This generates a
    pids directry, and stdout and stderr directories. Specify the number of model testers to start.

stop:
    Stop all model testers that have pid files in the pids_model_tester directory and delete the
    pids_model_tester directory.
"""


import os
import sys
import json
import subprocess
import time
import shutil

# The room in which lab pcs should be used for config generation.
ROOM = 2020
# The lowest pc id in the given room to use
PC_START = 10

# The base directory of this application. Remote commands will
# need to cd to this directory before running the python script
BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
# The directory to redirect the stdout to
STDOUT_DIR = 'stdout_model_trainer'
# The directory to redirect the stderr to
STDERR_DIR = 'stderr_model_trainer'
# The directory to put the pids file in
PID_DIR = 'pids_model_trainer'



# Start with PC_START and PORT_START
next_server = PC_START

def resolve_hostname(pcid):
    return f'dh{ROOM}pc{pcid:02}.utm.utoronto.ca'

def is_up(hostname):
    return 0 == os.system("ping -c 1 " + hostname + " > /dev/null 2>&1")

def next_free_server():
    global next_server

    while not is_up(resolve_hostname(next_server)):
        print(f"{next_server} is not up")
        next_server += 1
        if next_server >50:
            print("No more servers available.")
            sys.exit(1)
    if next_server > 50:
        print("No more servers available.")
        sys.exit(1)

    next_server += 1

    return resolve_hostname(next_server - 1)

# def run_remote(host, command, stdout="/dev/null", stderr="/dev/null"):
#     proc = subprocess.run(f"ssh {host} \"nohup sh -c '{command}' 1>{stdout} 2>{stderr} &\"",
#                        shell=True, capture_output=True, text=True)
#     return proc

def delete_directory(directory):
    try:
        # Check if the directory exists
        if os.path.exists(directory):
            # Delete the directory and its contents
            shutil.rmtree(directory)
            print(f"Directory '{directory}' and its contents have been deleted successfully.")
        else:
            print(f"Directory '{directory}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the directory: {e}")

def run(command, stdout="/dev/null", stderr="/dev/null", hangup=True):
    if not hangup:
        command = f"nohup sh -c '{command}' 1>{stdout} 2>{stderr} &"
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)

    return proc

def start_model_testers(num_model_trainers):
    """
    Start the model_trainers on the different lab machines. This generates a
    pids directry, and stdout and stderr directories.
    """
    try:
        os.mkdir(PID_DIR)
    except FileExistsError:
        print("ERROR: Pids directory still exists. Exiting.")
        sys.exit(1)

    try:
        os.mkdir(STDOUT_DIR)
    except FileExistsError:
        print("ERROR: Stdout directory still exists. Exiting.")
        sys.exit(1)

    try:
        os.mkdir(STDERR_DIR)
    except FileExistsError:
        print("ERROR: Stderr directory still exists. Exiting.")
        sys.exit(1)

    lst_num_hidden = [(2*i, 2 * (i + 1)) for i in range(1, num_model_trainers+1)]
    for i in range(num_model_trainers):
        host = next_free_server()
        print(f"Starting model_trainer on {host}")
        stdout = f"{BASE_DIR}/{STDOUT_DIR}/{host}.out"
        stderr = f"{BASE_DIR}/{STDERR_DIR}/{host}.err"
        pid_file = f"{BASE_DIR}/{PID_DIR}/{host}.pid"
        input_file = f"{BASE_DIR}/clean_dataset.csv"
        output_file = f"{BASE_DIR}/output_{i}.txt"
        command = f"cd {BASE_DIR}; python3 model_dist.py {lst_num_hidden[i][0]} {lst_num_hidden[i][1]} 20 300 {input_file} {output_file}"
        run(command, stdout, stderr, False)
        run(f"pgrep -U harisha1 -f 'python' > {pid_file}")


def stop_model_trainers():
    """
    Stop all model_trainers that have pid files in the pids_model_tester directory and delete the
    pids_model_tester directory.
    """
    for pidfile in os.listdir(PID_DIR):
        host = pidfile.split(".")[0]
        try: 
            with open(PID_DIR + "/" + pidfile) as f:
                pid = int(f.read())
                subprocess.run(f"ssh {host} \"kill {pid}\"",
                            shell=True, capture_output=True, text=True)
                print(f"Killed PID {pid} on {host}")
        except FileNotFoundError:
            print(f"Pid file {pidfile} does not exist.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: distribute_model.py <command> <args>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "start":
        if len(sys.argv) != 3:
            print("Usage: distribute_model.py start <num_model_trainers>")
            sys.exit(1)
        num_model_trainers = int(sys.argv[2])
        start_model_testers(num_model_trainers)
    
    elif command == "stop":
        stop_model_trainers()
        os.rmdir(PID_DIR)
        os.rmdir(STDOUT_DIR)
        os.rmdir(STDERR_DIR)
    
    else:
        print("Invalid command")
        sys.exit(1)

    



    

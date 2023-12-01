"""
Copyright 2021 AI-SPRINT

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from math import ceil
import csv
import os
from sys import argv


def read_partial_log(filename, step=-1):

    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    if step < 0:
        step = len(lines)

    t = 0

    # (id, type): time
    scheduled_tasks = {}
    executing_tasks = {}
    completed_tasks = {}

    # type: counter, total_time
    count_completed_tasks = {}

    tasks_ = []

    first_found = False
    first_executing_found = False

    start_clock = -1
    elapsed_time = 0

    for line in lines:

        if "NEW_TASK" in line:

            if not first_found:
                first_found = True
                start_clock = int(line.split(" ")[0])

            t = t + 1
            task_data = line.split("NEW_TASK ")[1].split(" ")[:2]

            scheduled_tasks[(int(task_data[0]), int(task_data[1]))] = int(line.split(" ")[0]) - start_clock
            elapsed_time = int(line.split(" ")[0]) - start_clock

        elif "EXECUTING_TASK" in line:

            if not first_executing_found:
                first_executing_found = True

            t = t + 1
            task_data = line.split("EXECUTING_TASK ")[1].split(" ")[:2]

            del scheduled_tasks[(int(task_data[0]), int(task_data[1]))]
            executing_tasks[(int(task_data[0]), int(task_data[1]))] = int(line.split(" ")[0]) - start_clock
            elapsed_time = int(line.split(" ")[0]) - start_clock

        elif "COMPLETED_TASK" in line:

            t = t + 1
            task_data = line.split("COMPLETED_TASK ")[1].split(" ")[:2]

            completed_tasks[(int(task_data[0]), int(task_data[1]))] = int(line.split(" ")[0]) - start_clock
            elapsed_time = int(line.split(" ")[0]) - start_clock

            if int(task_data[1]) not in count_completed_tasks:
                count_completed_tasks[int(task_data[1])] = [0, 0]
            count_completed_tasks[int(task_data[1])][0] += 1
            count_completed_tasks[int(task_data[1])][1] += completed_tasks[(int(task_data[0]), int(task_data[1]))] - executing_tasks[(int(task_data[0]), int(task_data[1]))]

            del executing_tasks[(int(task_data[0]), int(task_data[1]))]

        #elif "XXXXX" in line and "APP START" not in line and "SYNCHRONIZE" not in line and "FAILED_TASK" not in line and not "CANCELLED_TASK" in line and not "APP STOP" in line:
        elif "APP START" not in line and "SYNCHRONIZE" not in line and "FAILED_TASK" not in line and not "CANCELLED_TASK" in line and not "APP STOP" in line:
            import pdb; pdb.set_trace()

        if t == step and first_executing_found:

            #print("Scheduled tasks: ", len(scheduled_tasks))
            #print("Executing tasks: ", len(executing_tasks))
            #print("Completed tasks: ", len(completed_tasks))
            break
                

    task_header = ["ID", "COUNT", "AVG"]

    for id in count_completed_tasks:
        tasks_.append([id, count_completed_tasks[id][0], count_completed_tasks[id][1]/count_completed_tasks[id][0]])

    with open('partial_completitions.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(task_header)
        writer.writerows(tasks_)

    return elapsed_time


def read_exec_times_data(exec_time_file):
        
    exec_time_stats = {} # key: (ID, stat)
                
    header = True
    IDs = []
    with open(exec_time_file) as csvfile:
        data_csv = csv.reader(csvfile, delimiter=',')
        for row in data_csv:
            if not header:
                IDs.append(row[0])
                for idx in range(1,len(row)):
                    exec_time_stats[(row[0],header_line[idx])] = row[idx]
            else:
                header = False
                header_line = row 

    return exec_time_stats, IDs


def estimate_CPUs_number(current_log_file, reconfiguration_time, total_execution_time_constraint, profiling_data, min_cores, max_cores):

    # Analyse partial log
    elapsed_time = read_partial_log(current_log_file)
    print("Elapsed time: ", elapsed_time/1000, "s")
    partial_log_file = 'partial_completitions.csv'
    exec_times_partial, IDs_partial = read_exec_times_data(partial_log_file)
    os.remove(partial_log_file)

    # Load history
    exec_time_postprocessed, IDs = read_exec_times_data(profiling_data)

    # Count the remaining tasks
    remaining_tasks = {}

    for ID in IDs:

        if ID in IDs_partial:
            remaining_tasks[ID] = float(exec_time_postprocessed[(ID,'COUNT')]) - float(exec_times_partial[(ID,'COUNT')])
        else:
            remaining_tasks[ID] = float(exec_time_postprocessed[(ID,'COUNT')])

    # Compute residual time
    residual_time_real = total_execution_time_constraint - (elapsed_time/1000 + reconfiguration_time)
    
    # Dychotomic search
    residual_est_min_cores = sum(ceil(remaining_tasks[s])/min_cores*float(exec_time_postprocessed[s,'AVG']) for s in IDs)/1000
    residual_est_max_cores = sum(ceil(remaining_tasks[s])/max_cores*float(exec_time_postprocessed[s,'AVG']) for s in IDs)/1000

    if residual_time_real < residual_est_max_cores:

        print("Not possible to migrate application")
        return -1, elapsed_time
    
    if residual_time_real > residual_est_min_cores:

        print("Use the minimum number of cores")
        return min_cores, elapsed_time

    while max_cores - min_cores > 1:
        
        mid_cores = (min_cores + max_cores) // 2
        residual_est_mid_cores = sum(ceil(remaining_tasks[s])/mid_cores*float(exec_time_postprocessed[s,'AVG']) for s in IDs)/1000
        
        if residual_est_mid_cores < residual_time_real:
            max_cores = mid_cores
            residual_est_max_cores = residual_est_mid_cores
        else: 
            min_cores = mid_cores
            residual_est_min_cores = residual_est_mid_cores

        #print(min_cores, max_cores)

    if residual_time_real > residual_est_min_cores:
        return min_cores, elapsed_time
    return max_cores, elapsed_time


if __name__ == '__main__':

    current_log_file = argv[1]
    reconfiguration_time = float(argv[2])
    total_execution_time_constraint = float(argv[3])
    profiling_data = argv[4]
    min_cores = int(argv[5])
    max_cores = int(argv[6])
    
    cores_, elapsed_time = estimate_CPUs_number(current_log_file, reconfiguration_time, total_execution_time_constraint, profiling_data, min_cores, max_cores)

    print("Estimated core number:", cores_)
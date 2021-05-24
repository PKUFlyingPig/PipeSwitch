import subprocess
import os
import time
from experi_reproduce.throughput import ready_throughput, switch_throughput
from util.util import plot_grouped_barchart


batch_size = 8
scheduling_cycle = ['1','2','5','10','30']

def ready_server(model):
    p = subprocess.Popen(['python','../ready_model/ready_model.py',model]) 
    time.sleep(3)
    return p

def pipe_server():
    p = subprocess.Popen(['python','../pipeswitch/main.py','model_list.txt'])
    return p

def mps_server(model):
    p = subprocess.Popen(['python','./mps_server.py', model])
    return p

def kill_start_server():
    p = subprocess.Popen(['python','../kill_restart/kill_restart.py'])
    return p

def main():
    m = 'resnet152'
    #ready_model
    ready_model_throughput_list = dict.fromkeys(scheduling_cycle)
    for t in scheduling_cycle:
        p = ready_server(m)
        time.sleep(10) # wait for the server to load its model
        ready_model_throughput_list[t] = ready_throughput(m, batch_size, int(t))
        p.kill()
        print("-------------------")
        time.sleep(5)

    #pipeswitch
    pipeswitch_throughput_list = dict.fromkeys(scheduling_cycle)
    for t in scheduling_cycle:
        p = pipe_server()
        pipeswitch_throughput_list[t] = switch_throughput(m, batch_size, int(t))
        p.kill()
        print("-------------------")
        time.sleep(5)

    # kill_restart
    kill_restart_throughput_list = dict.fromkeys(scheduling_cycle)
    for t in scheduling_cycle:
        p = kill_start_server()
        time.sleep(10) # wait for the server to load its model
        kill_restart_throughput_list[t] = switch_throughput(m, batch_size,int(t))
        p.kill()
        print("-------------------")
        time.sleep(5)

    #MPS
    # start MPS daemon
    os.system("sudo nvidia-cuda-mps-control -d")
    time.sleep(5)

    MPS_throughput_list = dict.fromkeys(scheduling_cycle)
    for t in scheduling_cycle:
        p = mps_server(m)
        time.sleep(20) # wait for the server to start services and training task
        MPS_throughtput_list[t] = switch_throughput(m, batch_size, int(t))
        p.kill()
        print("-------------------")
        time.sleep(5)
    # shut down MPS daemon
    os.system("echo quit | nvidia-cuda-mps-control")
    time.sleep(5)

       
    grouped_data = {"Ready model":ready_model_throughput_list.values(),
                   "PipeSwitch":pipeswitch_throughput_list.values(),
                   "MPS":MPS_throughput_list.values(),
                   "Stop-and-start":kill_restart_throughput_list.values()}
    print(grouped_data)
    plot_grouped_barchart(scheduling_cycle, grouped_data, "throughput (batches/sec)", 0.15, "figure6a.png")
    print("figure 6a is done.")

if __name__ == "__main__":
    main()
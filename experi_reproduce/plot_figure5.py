"""
This script plots the 
"""
import subprocess
import time
from util.util import request_switch, request_infer, plot_cutted_grouped_barchart

model_names = ['resnet152', 'inception_v3', 'bert_base']
batch_size = 8


def ready_server(model):
    p = subprocess.Popen(['python','../ready_model/ready_model.py',model]) 
    time.sleep(3)
    return p

def pipe_server():
    p = subprocess.Popen(['python','../pipeswitch/main.py','model_list.txt'])
    return p

def kill_start_server():
    p = subprocess.Popen(['python','../kill_restart/kill_restart.py'])
    return p

def main():
    # #ready_model
    # ready_model_latency_list = dict.fromkeys(model_names)
    # for m in model_names:
    #     p = ready_server(m)
    #     mean_latency = request_infer(m, batch_size)
    #     p.kill()
    #     ready_model_latency_list[m] = mean_latency
    #     print("-------------------")
    #     time.sleep(5)
    # print(ready_model_latency_list)

    #pipeswitch
    # pipeswitch_latency_list = dict.fromkeys(model_names)
    # for m in model_names:
    #     p = pipe_server()
    #     time.sleep(30) # wait for the model to be loaded
    #     mean_latency = request_switch(m, batch_size)
    #     p.kill()
    #     pipeswitch_latency_list[m] = mean_latency
    #     print("-------------------")
    #     time.sleep(5)
    # print(pipeswitch_latency_list)

    #kill_restart
    # kill_restart_latency_list = dict.fromkeys(model_names)
    # for m in model_names:
    #     p = kill_start_server()
    #     mean_latency = request_switch(m, batch_size)
    #     p.kill()
    #     kill_restart_latency_list[m] = mean_latency
    #     print("-------------------")
    #     time.sleep(5)

    # # print("ready model : ", ready_model_latency_list)
    # # print("pipeswitch : ", pipeswitch_latency_list)
    # print("kill restart : ", kill_restart_latency_list)

    # plot figure
    ready_model_latency_list = dict.fromkeys(model_names, 40)
    pipeswitch_latency_list = dict.fromkeys(model_names, 50)
    mps_latency_list = dict.fromkeys(model_names, 300)
    kill_restart_latency_list = dict.fromkeys(model_names, 7000)
    grouped_data = {"Ready model":ready_model_latency_list.values(),
                    "PipeSwitch":pipeswitch_latency_list.values(),
                    "MPS":mps_latency_list.values(),
                    "Stop-and-start":kill_restart_latency_list.values()}
    plot_cutted_grouped_barchart(model_names, grouped_data, "Latency (ms)", 0.15, "figure5.png")
    print("figure 5 is done.")

if __name__ == '__main__':
    main()
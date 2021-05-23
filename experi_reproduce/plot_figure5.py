"""
This script plots the 
"""
import subprocess
import time
from util.util import request_new_inference, request_switch, request_infer, request_new_inference, plot_cutted_grouped_barchart

model_names = ['resnet152', 'inception_v3']
batch_size = 8


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
    # ready_model
    # ready_model_latency_list = dict.fromkeys(model_names)
    # for m in model_names:
    #    p = ready_server(m)
    #    time.sleep(10) # wait for the server to load its model
    #    mean_latency = request_infer(m, batch_size)
    #    p.kill()
    #    ready_model_latency_list[m] = mean_latency
    #    print("-------------------")
    #    time.sleep(5)

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

    # MPS (multi-process service)
    MPS_latency_list = dict.fromkeys(model_names)
    for m in model_names:
        p = mps_server(m)
        time.sleep(30) # wait for the server to start services and training task
        mean_latency = request_new_inference(m, batch_size)
        p.kill()
        MPS_latency_list[m] = mean_latency
        print("-------------------")
        time.sleep(5)


    

    #kill_restart
    # kill_restart_latency_list = dict.fromkeys(model_names)
    # p = kill_start_server()
    # time.sleep(10) # wait for the server to load its model
    # for m in model_names:
    #    mean_latency = request_switch(m, batch_size)
    #    kill_restart_latency_list[m] = mean_latency
    #    print("-------------------")
    #    time.sleep(5)

    # print("ready model : ", ready_model_latency_list)
    # #print("pipeswitch : ", pipeswitch_latency_list)
    print("MPS : ", MPS_latency_list)
    # print("kill restart : ", kill_restart_latency_list)

    # plot figure
    #ready_model_latency_list = dict.fromkeys(model_names, 40)
    #pipeswitch_latency_list = dict.fromkeys(model_names, 50)
    #mps_latency_list = dict.fromkeys(model_names, 300)
    #kill_restart_latency_list = dict.fromkeys(model_names, 7000)
    #grouped_data = {"Ready model":ready_model_latency_list.values(),
    #                "PipeSwitch":pipeswitch_latency_list.values(),
    #                "MPS":mps_latency_list.values(),
    #                "Stop-and-start":kill_restart_latency_list.values()}
    #plot_cutted_grouped_barchart(model_names, grouped_data, "Latency (ms)", 0.15, "figure5.png")
    #print("figure 5 is done.")

if __name__ == '__main__':
    main()
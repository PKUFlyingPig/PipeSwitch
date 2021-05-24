"""
This script plots the figure 7 in paper
"""
import subprocess
import time
from util.util import request_switch, plot_grouped_barchart

model_names = ['resnet152', 'inception_v3', 'bert_base']
batch_size = 8


def pipe_server():
    p = subprocess.Popen(['python','../pipeswitch/main.py','model_list.txt'])
    return p

def main():
    #pipeswitch
    pipeswitch_latency_list = dict.fromkeys(model_names)
    for m in model_names:
        p = pipe_server()
        time.sleep(30) # wait for the model to be loaded
        mean_latency = request_switch(m, batch_size)
        p.kill()
        pipeswitch_latency_list[m] = mean_latency
        print("-------------------")
        time.sleep(5)

    print("pipeswitch : ", pipeswitch_latency_list)
    # plot figure
    grouped_data = {"PipeSwitch":pipeswitch_latency_list.values()}
    plot_grouped_barchart(model_names, grouped_data, "Latency (ms)", 0.15, "figure7.png")
    print("figure 7 is done.")

if __name__ == '__main__':
    main()

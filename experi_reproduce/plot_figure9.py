
import sys
import queue
import struct
import threading
import importlib
import time

import torch
import torch.multiprocessing as mp

from task.helper import get_model, get_data
from util.util import TcpServer, TcpAgent, plot_cutted_grouped_barchart, timestamp, plot_grouped_barchart
import statistics

model_names = ['resnet152', 'inception_v3', 'bert_base']
batch_size = 8

def one_process_switch(model_name):
    # Training Task
    # Load model
    task_name_train = '%s_training' % model_name
    model_module = importlib.import_module('task.' + task_name_train)
    model, func, _ = model_module.import_task()
    data_loader = model_module.import_data_loader()
    # Model to GPU
    model = model.to('cuda')
    timestamp('server', "move model to GPU")
    # Compute
    train_output = func(model, data_loader)

    # Inference Task reuses the same CUDA context
    data = get_data(model_name, batch_size)
    data_b = data.numpy().tobytes()
    # Load model
    model, func = get_model(model_name)
    time1 = time.time()
     # Model to GPU
    model = model.eval().cuda()
    # Compute
    infer_output = func(model, data_b)
    time2 = time.time()
    return (time2 - time1)*1000

def main():
    # oneprocess_latency_list = dict.fromkeys(model_names)
    # for m in model_names:
    #     latency_list = []
    #     for _ in range(2):
    #         latency_list.append(one_process_switch(m))
    #     oneprocess_latency_list[m] = statistics.mean(latency_list)
    #     print(oneprocess_latency_list)

    oneprocess_latency_list = {'resnet152': 127, 'inception_v3': 75, 'bert_base': 196}
    pipeswitch_latency_list = {'resnet152': 63.13779322306315, 'inception_v3': 55.38988420698378, 'bert_base': 85}
    twoprocess_latency_list = {'resnet152': 6942.613863945007, 'inception_v3': 8037.053346633911, 'bert_base': 7500}
    grouped_data = {"PipeSwitch":pipeswitch_latency_list.values(),
                    "One Process":oneprocess_latency_list.values(),
                    "Two Process":twoprocess_latency_list.values()}
    plot_cutted_grouped_barchart(model_names, grouped_data, "Latency (ms)", 0.15, "figure9.png", 250, 6000)
    print("figure 9 is done.")

if __name__ == '__main__':
    main()

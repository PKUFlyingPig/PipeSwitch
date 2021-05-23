import sys
import queue
import struct
import threading
import importlib

import torch.multiprocessing as mp

from util.util import TcpServer, TcpAgent, timestamp

def func_get_request(qout):
    # Listen connections
    server = TcpServer('localhost', 12345)
    timestamp("server", "mps server up")
    while True:
        # Get connection
        conn, _ = server.accept()
        agent = TcpAgent(conn)

        model_name_length_b = agent.recv(4)
        model_name_length = struct.unpack('I', model_name_length_b)[0]
        if model_name_length == 0:
            break
        model_name_b = agent.recv(model_name_length)
        model_name = model_name_b.decode()
        timestamp('tcp', 'get_name')

        data_length_b = agent.recv(4)
        data_length = struct.unpack('I', data_length_b)[0]
        if data_length > 0:
            data_b = agent.recv(data_length)
        else:
            data_b = None
        timestamp('tcp', 'get_data')
        qout.put((agent, model_name, data_b))

def func_schedule(qin):
    while True:
        agent, model_name, data_b = qin.get()
        active_worker = mp.Process(target=worker_compute, args=(agent, model_name, data_b))
        active_worker.start()

def worker_compute(agent, model_name, data_b):
    # Load model
    model_module = importlib.import_module('task.' + model_name)
    model, func, _ = model_module.import_task()
    data_loader = model_module.import_data_loader()

    # Model to GPU
    model = model.to('cuda')

    # Compute
    if 'training' in model_name:
        agent.send(b'FNSH')
        del agent
        timestamp('server', 'reply')

        output = func(model, data_loader)
        timestamp('server', 'complete')

    else:
        output = func(model, data_b)
        timestamp('server', 'complete')

        agent.send(b'FNSH')
        del agent
        timestamp('server', 'reply')
    
def training_task(model_name):
    # Load model
    model_module = importlib.import_module('task.' + model_name)
    model, func, _ = model_module.import_task()
    data_loader = model_module.import_data_loader()

    timestamp('server', 'start training task')
    # Model to GPU
    model = model.to('cuda')

    timestamp('server', "move model to GPU")

    # Compute
    output = func(model, data_loader)
    

def main():
    # Get model name
    model_name = sys.argv[1]

    # Create threads and worker process
    q_to_schedule = queue.Queue()
    t_get = threading.Thread(target=func_get_request, args=(q_to_schedule,))
    t_get.start()
    t_schedule = threading.Thread(target=func_schedule, args=(q_to_schedule,))
    t_schedule.start()

    # create training_task
    task_name_train = '%s_training' % model_name
    train = threading.Thread(target=training_task, args=(task_name_train,))
    train.start()

    # Accept connection
    t_get.join()
    t_schedule.join()
    train.join()
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

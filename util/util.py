import sys
import time
import socket

import struct
import statistics
import matplotlib.pyplot as plt
import numpy as np

from task.helper import get_data

def request_infer(model_name, batch_size):
    """
    In each iteration, send an inferece request, wait for the reply and record the latency without switching.
    assuming the model has already been loaded into GPU.
    return the average latency (the time between the client send the inference request and the client receive the reply)
    """
    # Load image
    data = get_data(model_name, batch_size)

    latency_list = []
    for _ in range(100):
        timestamp('client', 'before_request')

        # Connect
        client = TcpClient('localhost', 12345)
        timestamp('client', 'after_connect')
        time_1 = time.time()

        # Serialize data
        task_name = model_name + '_inference'
        task_name_b = task_name.encode()
        task_name_length = len(task_name_b)
        task_name_length_b = struct.pack('I', task_name_length)
        data_b = data.numpy().tobytes()
        length = len(data_b)
        length_b = struct.pack('I', length)
        timestamp('client', 'after_serialization')

        # Send Data
        client.send(task_name_length_b)
        client.send(task_name_b)
        client.send(length_b)
        client.send(data_b)
        timestamp('client', 'after_send')

        # Get reply
        reply_b = client.recv(4)
        reply = reply_b.decode()
        if reply == 'FAIL':
            timestamp('client', 'FAIL')
            break
        timestamp('client', 'after_reply')
        time_2 = time.time()

        model_name_length = 0
        model_name_length_b = struct.pack('I', model_name_length)
        client.send(model_name_length_b)
        timestamp('client', 'close_training_connection')

        timestamp('**********', '**********')
        latency = (time_2 - time_1) * 1000
        latency_list.append(latency)
        
    stable_latency_list = latency_list[10:]
    return statistics.mean(stable_latency_list)

def request_new_inference(model_name, batch_size):
    """
    In each iteration, send an inferece request, wait for the reply and record the latency without switching.
    assuming the model has not already been loaded into GPU.
    return the average latency (the time between the client send the inference request and the client receive the reply)
    """
    task_name_inf = '%s_inference' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    latency_list = []
    for _ in range(20):
        # Connect
        client_inf = TcpClient('localhost', 12345)
        timestamp('client', 'after_inference_connect')
        time_1 = time.time()

        # Send inference request
        send_request(client_inf, task_name_inf, data)

        # Recv inference reply
        recv_response(client_inf)
        time_2 = time.time()
        latency = (time_2 - time_1) * 1000
        latency_list.append(latency)

        time.sleep(1)
        close_connection(client_inf)
        time.sleep(1)
        timestamp('**********', '**********')

    stable_latency_list = latency_list[10:]
    return statistics.mean(stable_latency_list)

def send_request(client, task_name, data):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize data
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    if data is not None:
        data_b = data.numpy().tobytes()
        length = len(data_b)
    else:
        data_b = None
        length = 0
    length_b = struct.pack('I', length)
    timestamp('client', 'after_inference_serialization')

    # Send Data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(length_b)
    if data_b is not None:
        client.send(data_b)
    timestamp('client', 'after_request_%s' % task_name)

def recv_response(client):
    reply_b = client.recv(4)
    reply = reply_b.decode()
    timestamp('client', 'after_reply')

def close_connection(client):
    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_connection')

def request_switch(model_name, batch_size):
    """
    In each iteration, send a training request, then send an inference request to record the latency with switching.
    return the average latency. (the time between the client send the inference request and the client receive the reply)
    """
    task_name_inf = '%s_inference' % model_name
    task_name_train = '%s_training' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    latency_list = []
    for _ in range(20):
        # Send training request
        client_train = TcpClient('localhost', 12345)
        send_request(client_train, task_name_train, None)
        time.sleep(4)

        # Connect
        client_inf = TcpClient('localhost', 12345)
        timestamp('client', 'after_inference_connect')
        time_1 = time.time()

        # Send inference request
        send_request(client_inf, task_name_inf, data)

        # Recv inference reply
        recv_response(client_inf)
        time_2 = time.time()
        latency = (time_2 - time_1) * 1000
        latency_list.append(latency)

        time.sleep(1)
        recv_response(client_train)
        close_connection(client_inf)
        close_connection(client_train)
        time.sleep(1)
        timestamp('**********', '**********')

    stable_latency_list = latency_list[10:]
    return statistics.mean(stable_latency_list)

def plot_grouped_barchart(groups, grouped_data, ylabel, bar_width, figurename):
    ngroup = len(groups)
    x = np.arange(ngroup)  # the label locations
    fig, ax = plt.subplots()
    startx = x - (ngroup - 1) / 2 * bar_width
    rects = []
    for i, (label, data) in enumerate(grouped_data.items()):
    	rects.append(ax.bar(startx + i * bar_width, data, bar_width, label=label))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()


    fig.tight_layout()
    fig.savefig(figurename)

def plot_cutted_grouped_barchart(groups, grouped_data, ylabel, bar_width, figurename):
    ngroup = len(groups)
    x = np.arange(ngroup)  # the label locations
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    startx = x - (ngroup - 1) / 2 * bar_width
    rects = []
    for i, (label, data) in enumerate(grouped_data.items()):
    	rects.append(ax1.bar(startx + i * bar_width, data, bar_width, label=label))
    	rects.append(ax2.bar(startx + i * bar_width, data, bar_width, label=label))

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylim(6000, 10000)
    ax2.set_ylim(0, 1300)
    ax2.set_ylabel(ylabel)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.legend()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.tight_layout()
    fig.savefig(figurename)

def timestamp(name, stage):
    print ('TIMESTAMP, %s, %s, %f' % (name, stage, time.time()), file=sys.stderr)

class TcpServer():
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.address, self.port))
        self.sock.listen(1)

    def __del__(self):
        self.sock.close()

    def accept(self):
        conn, address = self.sock.accept()
        return conn, address


class TcpAgent:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.conn.close()

    def send(self, msg):
        self.conn.sendall(msg)

    def recv(self, msg_len):
        return self.conn.recv(msg_len, socket.MSG_WAITALL)

    def settimeout(self, t):
        self.conn.settimeout(t)


class TcpClient(TcpAgent):
    def __init__(self, address, port):
        super().__init__(None)
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((address, port))

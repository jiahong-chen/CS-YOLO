import os
import sys
import cv2
import darknet
import argparse
import threading
import numpy as np

from msvcrt import getch
from socketIO import Socket
# from classifier import Classifier


def handle_client(server, network, class_names):
    image = server.recieve_frame()
    coordinates = image_detection(image, network, class_names, args.thresh)
    server.send(coordinates)


def keyboard_event():
    while True:
        if ord(getch()) in [27, 113]:
            raw_input = input('你想要結束本服務嗎(yes/no)?')
            if raw_input in ['y', 'Y', 'yes', 'YES']:
                os._exit(0) 
            else:
                continue


def image_detection(image, network, class_names, thresh):
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(image)
    
    darknet_image = darknet.make_image(net_width, net_height, 3)

    h, w, _ = image.shape
    image_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.cuda.resize(image_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.download().tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    coordinates = list()
    # pass_list = ['no_hat']
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox, (net_width/w, net_height/h))
        # if not label in pass_list:
        #     label, confidence = classifer.classifiercation(image_rgb[top:bottom, left:right])
        coordinates.append((label, confidence, (left, top, right, bottom)))
    return coordinates


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    # YOLO object tracking
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    # CNN classification
    parser.add_argument("--arch", type=str, default='vgg19', help="CNN architecture")
    parser.add_argument("--model_path", type=str, default='./model/model_best.pth.tar',
                        help="CNN training model path")
    parser.add_argument("--label_path", type=str, default='./label/obj,txt',
                        help="classification of label")
    # build server service
    parser.add_argument("--ip", "-i", type=str, default='127.0.0.1')
    parser.add_argument("--port", "-p", type=int, default=9999)
    args = parser.parse_args()

    # classifer = Classifier(args.arch, args.model_path, args.label_path)
    threading.Thread(target=keyboard_event).start()
    network, class_names, _ = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    net_width = darknet.network_width(network)
    net_height = darknet.network_height(network)

    server = Socket(args.ip, args.port)
    server.startServer()
    while True:
        server.client_conn, server.client_addr = server.socket.accept()
        handle_client(server, network, class_names)
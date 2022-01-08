import os
import sys
import cv2
import pickle
import darknet
import logging
import argparse
import threading
import numpy as np

from msvcrt import getch
from socketIO import Socket
from classifier import Classifier

def parser():
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
    return parser.parse_args()


def handle_client(server, network, class_names):
    print(f"[new connection] {server.client_addr} connected.")
    image = server.recieve()
    coordinates = image_detection(image, network, class_names, args.thresh)
    data=pickle.dumps(coordinates)
    server.client_conn.send(data)


def keyboard_event():
    while True:
        if ord(getch()) in [27, 113]:
            raw_input = input('你想要結束本服務嗎(yes/no)?')
            if raw_input in ['y', 'Y', 'yes', 'YES']:
                os._exit(0) 
            else:
                continue


def image_detection(image, network, class_names, thresh):
    net_width = darknet.network_width(network)
    net_height = darknet.network_height(network)
    darknet_image = darknet.make_image(net_width, net_height, 3)

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    coordinates = list()
    pass_list = ['no_hat']
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox, (net_width/w, net_height/h))
        if not label in pass_list:
            label, confidence = classifer.classifiercation(image_rgb[top:bottom, left:right])
        coordinates.append((label, confidence, (left, top, right, bottom)))
    return coordinates


def main():
    global classifer

    classifer = Classifier(args.arch, args.model_path, args.label_path)
    threading.Thread(target=keyboard_event).start()
    network, class_names, _ = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )

    logger = logging.getLogger('Server')
    logger.setLevel(logging.DEBUG)

    server = Socket(args.ip, args.port)
    server.startServer()
    while True:
        server.client_conn, server.client_addr = server.socket.accept()
        logging.info(f'[Info] Connected to: {server.client_addr[0]}')
        threading.Thread(target=handle_client, args=(server, network, class_names,)).start()

if __name__ == '__main__':
    global args
    args = parser()
    main()
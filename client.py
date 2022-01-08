import os
import cv2
import time
import pickle
import random
import darknet
import argparse
import numpy as np
import multiprocessing

from queue import Queue
from socketIO import Socket
from threading import Thread
from functools import partial
from multiprocessing.pool import ThreadPool

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    parser.add_argument("--classes", type=str)
    return parser.parse_args()


def get_colors(args):
    colors = dict()
    with open(args.classes, 'r') as f:
        for line in f:
            colors[line.split('\n')[0]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors


def socket_handler(host, port, frame):
    client = Socket(host, port)
    client.startClient()
    client.send(frame)
    coordinates = client.recieve_text()
    client.close()
    return coordinates


def video_capture(original_frame_queue, yuv_frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        original_frame_queue.put(frame)
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv_frame)
        new_y = cv2.equalizeHist(y)
        frame = cv2.cvtColor(cv2.merge([new_y, u, v]), cv2.COLOR_YUV2BGR)
        yuv_frame_queue.put(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, (net_width, net_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(net_width, net_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(original_frame_queue, yuv_frame_queue, detections_queue):
    while cap.isOpened():
        original_frame = original_frame_queue.get()
        yuv_frame = yuv_frame_queue.get()
        detections = detections_queue.get()
        h, w, _ = original_frame.shape
        prev_time = time.time()
        if original_frame is not None:
            pool = ThreadPool(processes=multiprocessing.cpu_count()-1)
            server_detections = list()
            images, coordinates = cut_images(detections, yuv_frame, (net_width/w, net_height/h))
            for id_, ((hat_image, hook_image), (hat_coord, hook_coord)) in enumerate(zip(images, coordinates)):
                ip = [('127.0.0.1', 9000, hat_image), ('127.0.0.1', 9999, hook_image)]
                func = partial(socket_handler)
                data = pool.starmap_async(func, ip).get()
                for category in data:
                    for detection in category:
                        if 'hat' in detection[1]:
                            server_detections.append((id_,)+detection+(hat_coord,))
                        if 'hook' in detection[1]:
                            server_detections.append((id_,)+detection+(hook_coord,))
            pool.close()
            pool.join()
            print('FPS:{}'.format(1/(time.time()-prev_time)))
            show_frame = draw_person_boxes(detections, original_frame, (net_width/w, net_height/h))
            show_frame = draw_object_boxes(server_detections, show_frame)
            cv2.imshow('image', show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                os._exit(0)
    cap.release()
    cv2.destroyAllWindows()


def cut_images(detections, yuv_frame, zoom):
    images = list()
    coordinates = list()
    h, w, _ = yuv_frame.shape
    for _, _, bbox in detections:
        left, top, right, bottom = bbox2points(bbox, zoom)
        hat_image, hat_coord = cut_hat_image(yuv_frame, h, w, left, top, right, bottom)
        hook_image, hook_coord = cut_hook_image(yuv_frame, h, w, left, top, right, bottom)
        images.append((hat_image, hook_image))
        coordinates.append((hat_coord, hook_coord))
    return (images, coordinates)

def cut_hat_image(frame, h, w, left, top, right, bottom):
    left = left if left >= 0 else 0
    top = top if top >= 0 else 0
    right = right if right <= w else w
    bottom = bottom if bottom <= h else h
    return frame[top:bottom, left:right], (left, top)

def cut_hook_image(frame, h, w, left, top, right, bottom):
    box_h = bottom-top
    box_w = right-left
    prop = (right-left)/(bottom-top)

    if prop > 0.8:
        top = int(top-box_h if top-box_h >=0 else 0)
    else:
        top = int(top-box_h*0.2 if top-box_h*0.2 >=0 else 0)

    left = int(left-box_w*0.9 if left-box_w*0.9 >= 0 else 0)
    right = int(right+box_w*0.9 if right+box_w*0.9<=w else w)
    bottom = int(bottom+box_h*0.2 if bottom+box_h*0.2 >= h else h)
    return frame[top:bottom, left:right], (left, top)

def bbox2points(bbox, zoom):
    zoom_x, zoom_y = zoom
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2))/zoom_x)
    xmax = int(round(x + (w / 2))/zoom_x)
    ymin = int(round(y - (h / 2))/zoom_y)
    ymax = int(round(y + (h / 2))/zoom_y)
    return xmin, ymin, xmax, ymax


def resize_points(image_coordinates, bbox_coordinates):
    left, top, right, bottom = bbox_coordinates
    left = left+image_coordinates[0]
    top = top+image_coordinates[1]
    right = right+image_coordinates[0]
    bottom = bottom+image_coordinates[1]
    return left, top, right, bottom


def draw_person_boxes(detections, image, zoom):
    for (label, confidence, bbox) in detections:
        left, top, right, bottom = bbox2points(bbox, zoom)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def draw_object_boxes(detections, image):
    for _, label, confidence, bbox_coordinates, image_coordinates in detections:
        left, top, right, bottom = resize_points(image_coordinates, bbox_coordinates)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


if __name__ == '__main__':
    original_frame_queue = Queue()
    yuv_frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)

    args = parser()
    colors = get_colors(args)
    network, class_names, _ = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
    )

    net_width = darknet.network_width(network)
    net_height = darknet.network_height(network)
    cap = cv2.VideoCapture('AnochinstrapBhook-04.mp4')
    # set thread
    cap_thread = Thread(target=video_capture, args=(original_frame_queue, yuv_frame_queue, darknet_image_queue,))
    inference_thread = Thread(target=inference, args=(darknet_image_queue, detections_queue,))
    drawing_thread = Thread(target=drawing, args=(original_frame_queue, yuv_frame_queue, detections_queue,))
    # set daemon
    inference_thread.setDaemon(True)
    drawing_thread.setDaemon(True)  
    # start thread
    cap_thread.start()
    inference_thread.start()
    drawing_thread.start()
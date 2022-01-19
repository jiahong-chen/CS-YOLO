import os
import cv2
import time
import uuid
import random
import darknet
import logging
import argparse
import numpy as np
import multiprocessing as mp

from typing import List, Dict
from msvcrt import getch, kbhit
from socketIO import Socket
from threading import Thread
from functools import partial
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Queue, Lock

@dataclass
class RtspProxy():
    camera: List[str]
    original_image_queue: Queue
    yuv_image_queue: Queue
    detections_queue: Queue
    
    def __post_init__(self) -> None:
        self.lock = Lock()
        self.processes = list()
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def run(self) -> None:
        for idx, url in enumerate(self.camera):
            self.processes.append(Process(target=cap_camera_frame, name=f'process-{idx}', 
                                        args=(url, self.original_image_queue, self.yuv_image_queue, self.lock,)))
        for process in self.processes:
            logging.info('[Info] Starting process...')
            process.start()
            logging.info('[Info] Start process finished. pid: {}'.format(process.name))

    def stop(self) -> None:
        for process in self.processes:
            process.join()
            logging.info('[Info]: Joined process successfully!') 

    def restart(self, name) -> None:
        _, idx = name.split('-')
        self.processes.append(Process(target=cap_camera_frame, name=f'process-{idx}', 
                                        args=(self.camera[int(idx)], self.original_image_queue, self.yuv_image_queue, self.lock,)))
        
        logging.info('[Info] Restarting process...')
        self.processes[-1].start()
        logging.info('[Info] Restarting process finished. pid: {}'.format(self.processes[-1].name))

    def interrupt(self) -> None:
        for process in self.processes:
            logging.info('[Info]: Terminating slacking process. pid: {}'.format(process.name))
            process.terminate()
            time.sleep(0.5)
            if not process.is_alive():
                logging.info('[Info]: Process is a goner.')
                process.join(timeout=1.0)
                logging.info('[Info]: Joined process successfully!') 
            else:
                logging.info('[Info]: Joined process not successfully!') 
        self.original_image_queue.close()
        self.yuv_image_queue.close()
        self.detections_queue.close()
    
    def kill(self, process) -> None:
        process.terminate()
        time.sleep(0.5)
        if not process.is_alive():
            process.join(timeout=1.0)
            logging.info('[Info]: Killed process successfully!')


def watch_dog(rtsp) -> None:
    while True:
        for idx, process in enumerate(rtsp.processes):
            if not process.is_alive():
                rtsp.restart(process.name)
                del rtsp.processes[idx]

        if kbhit():
            key = ord(getch())
            if key in [27, 113]:
                raw_input = input('你想要結束本服務嗎(yes/no)? ')
                if raw_input in ['y', 'Y', 'yes', 'YES']:
                    rtsp.interrupt()
            if key == 107:
                raw_input = input('輸入你想結束的程序名稱(pid). ')
                for process in rtsp.processes:
                    if process.name == raw_input:
                        rtsp.kill(process)
                    else:
                        print(f'{raw_input} 不存在.')
        time.sleep(1)


def cap_camera_frame(url:str, original_image_queue:Queue, yuv_image_queue:Queue, lock: Lock) -> None:
    camera = cv2.VideoCapture(url)
    fpsLimit = 0.002
    st = time.time()
    while camera.isOpened():
        ret, frame = camera.read()
        curr = time.time()
        if (curr - st) > fpsLimit:
            if not ret:
                break
            try:
                lock.acquire()
                original_image_queue.put(frame)
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                yuv_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2YUV)
                y, u, v = cv2.cuda.split(yuv_frame)
                new_y = cv2.cuda.equalizeHist(y)
                gpu_yuv = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)
                cv2.cuda.merge([new_y, u, v], gpu_yuv)
                yuv_image_queue.put(cv2.cuda.cvtColor(gpu_yuv, cv2.COLOR_YUV2BGR).download())
            finally:
                lock.release()
            st = time.time()
    camera.release()


def socket_handler(host:str, port:int, frame:np.ndarray) -> List[tuple]:
    client = Socket(host, port)
    client.startClient()
    client.send(frame)
    coordinates = client.recieve_text()
    client.close()
    return coordinates


def get_colors() -> Dict[str, tuple]:
    colors = dict()
    with open(args.classes, 'r') as f:
        for line in f:
            colors[line.split('\n')[0]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return colors


def inference(yuv_image_queue:Queue, detections_queue:Queue) -> None:
    while True:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(yuv_image_queue.get(timeout=1))
        frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.cuda.resize(frame, (net_width, net_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(net_width, net_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.download().tobytes())
        detections = darknet.detect_image(network, class_names, img_for_detect, thresh=args.thresh)
        detections_queue.put(detections)
        darknet.free_image(img_for_detect)


def drawing(original_image_queue:Queue, yuv_image_queue:Queue, detections_queue:Queue) -> None:
    while True:
        prev_time = time.time()
        original_frame = original_image_queue.get(timeout=1)
        yuv_frame = yuv_image_queue.get(timeout=1)
        detections = detections_queue.get(timeout=1)
        h, w, _ = original_frame.shape
        # pool = ThreadPool(processes=mp.cpu_count()-1)
        # server_detections = list()
        # images, coordinates = cut_images(detections, yuv_frame, (net_width/w, net_height/h))
        # for id_, ((hat_image, hook_image), (hat_coord, hook_coord)) in enumerate(zip(images, coordinates)):
        #     ip = [('127.0.0.1', 9000, hat_image)]
        #     func = partial(socket_handler)
        #     data = pool.starmap_async(func, ip).get()
        #     for category in data:
        #         for detection in category:
        #             server_detections.append((id_,)+detection+(hat_coord,))
        #             # if 'hat' in detection[1]:
        #             #     server_detections.append((id_,)+detection+(hat_coord,))
        #             # if 'hook' in detection[1]:
        #             #     server_detections.append((id_,)+detection+(hook_coord,))
        # pool.close()
        # pool.join()
        show_frame = draw_person_boxes(detections, original_frame, (net_width/w, net_height/h))
        # show_frame = draw_object_boxes(server_detections, show_frame)
        print('FPS:{}'.format(1/(time.time()-prev_time)))
        # cv2.imshow('img', yuv_frame)
        cv2.imwrite(f'./tmp/{str(uuid.uuid4())}.jpg', show_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     os._exit(0)


def cut_images(detections:tuple, yuv_frame:np.ndarray, zoom:tuple) -> tuple:
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


def cut_hat_image(frame:np.ndarray, h:int, w:int, left:int, top:int, right:int, bottom:int) -> np.ndarray:
    left = left if left >= 0 else 0
    top = top if top >= 0 else 0
    right = right if right <= w else w
    bottom = bottom if bottom <= h else h
    return frame[top:bottom, left:right], (left, top)


def cut_hook_image(frame:np.ndarray, h:int, w:int, left:int, top:int, right:int, bottom:int) -> np.ndarray:
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


def bbox2points(bbox:tuple, zoom:tuple) -> tuple:
    zoom_x, zoom_y = zoom
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2))/zoom_x)
    xmax = int(round(x + (w / 2))/zoom_x)
    ymin = int(round(y - (h / 2))/zoom_y)
    ymax = int(round(y + (h / 2))/zoom_y)
    return (xmin, ymin, xmax, ymax)


def resize_points(image_coordinates:tuple, bbox_coordinates:tuple) -> tuple:
    left, top, right, bottom = bbox_coordinates
    left = left+image_coordinates[0]
    top = top+image_coordinates[1]
    right = right+image_coordinates[0]
    bottom = bottom+image_coordinates[1]
    return (left, top, right, bottom)


def draw_person_boxes(detections:tuple, image:np.ndarray, zoom:tuple) -> np.ndarray:
    for (label, confidence, bbox) in detections:
        left, top, right, bottom = bbox2points(bbox, zoom)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def draw_object_boxes(detections:tuple, image:np.ndarray) -> np.ndarray:
    for _, label, confidence, bbox_coordinates, image_coordinates in detections:
        left, top, right, bottom = resize_points(image_coordinates, bbox_coordinates)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


if __name__ == '__main__':
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
    args = parser.parse_args()

    original_image_queue = Queue()
    yuv_image_queue = Queue()
    detections_queue = Queue(maxsize=1)
    # set rtsp server
    # camera = ['rtsp://pkdemo.bovia.com.tw/roy_ptz?&key=tsmc_f15' for _ in range(4)]
    # camera = ['./AnochinstrapBhook-04.mp4' for _ in range(4)]
    camera = [f'test{idx}.mp4' for idx in range(1, 5)]
    rtsp = RtspProxy(camera, original_image_queue, yuv_image_queue, detections_queue)

    #set object detection
    colors = get_colors()
    network, class_names, _ = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
    )

    net_width = darknet.network_width(network)
    net_height = darknet.network_height(network)

    inference_thread = Thread(target=inference, args=(yuv_image_queue, detections_queue,))
    drawing_thread = Thread(target=drawing, args=(original_image_queue, yuv_image_queue, detections_queue,))
    watch_dog_thread = Thread(target=watch_dog, args=(rtsp,))
    # set daemon
    inference_thread.setDaemon(True)
    drawing_thread.setDaemon(True)
    watch_dog_thread.setDaemon(True)
    # start process & thread
    rtsp.run()
    inference_thread.start()
    drawing_thread.start()
    watch_dog_thread.start()
    # stop process
    rtsp.stop()
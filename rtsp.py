import os
from re import T
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
    mode: str
    camera: List[str]
    original_image_queue: Queue
    yuv_image_queue: Queue
    detections_queue: Queue
    
    def __post_init__(self) -> None:
        self.lock = Lock()
        self.processes = list()
        self.queues = [self.original_image_queue, self.yuv_image_queue, self.detections_queue]
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.run()

    def cap_camera_frame(self, url:str, original_image_queue:Queue, yuv_image_queue:Queue, lock:Lock) -> None:
        camera = cv2.VideoCapture(url)
        fpsLimit = 0.1
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

    def run(self) -> None:
        for idx, url in enumerate(self.camera):
            self.processes.append(Process(target=self.cap_camera_frame, name=f'process-{idx}', 
                                        args=(url, self.original_image_queue, self.yuv_image_queue, self.lock,)))
        for process in self.processes:
            logging.info('[Info] Starting process...')
            process.start()
            logging.info('[Info] Start process finished. pid: {}'.format(process.name))
        
        watch_dog_thread = Thread(target=self.watch_dog, args=(self.mode, self.processes,))
        watch_dog_thread.setDaemon(True)
        watch_dog_thread.start()

    def stop(self) -> None:
        for process in self.processes:
            process.join(timeout=1.0)
            logging.info('[Info]: Joined process successfully!') 
        
    def interrupt(self) -> None:
        for queue in self.queues:
            queue.close()
            queue.join_thread()
        for process in self.processes:
            logging.info('[Info]: Terminating slacking process. pid: {}'.format(process.name))
            process.terminate()
            time.sleep(1)
            if not process.is_alive():
                logging.info('[Info]: Process is a goner.')
                process.join(timeout=1.0)
                logging.info('[Info]: Joined process successfully!') 
            else:
                logging.info('[Info]: Joined process not successfully!') 

    def kill(self, process) -> None:
        process.terminate()
        time.sleep(1)
        if not process.is_alive():
            process.join(timeout=1.0)
            logging.info('[Info]: Killed process successfully!')

    def watch_dog(self, mode, processes) -> None:
        while True:
            if mode == 'stream':
                for idx, process in enumerate(processes):
                    if not process.is_alive():
                        logging.error(f'Process-{idx} was hang out.')
                        del processes[idx]
            if kbhit():
                key = ord(getch())
                if key in [27, 113]:
                    raw_input = input('你想要結束本服務嗎(yes/no)? ')
                    if raw_input in ['y', 'Y', 'yes', 'YES']:
                        self.interrupt()
                if key == 107:
                    raw_input = input('輸入想結束的程序名稱(pid). ')
                    for process in processes:
                        if process.name == raw_input:
                            self.kill(process)
                        else:
                            print(f'{raw_input} 不存在.')
            time.sleep(1)


@dataclass
class Detection:
    original_image_queue: Queue
    yuv_image_queue: Queue
    detections_queue: Queue

    def __post_init__(self):
        self.lock1 = Lock()
        self.lock2 = Lock()
        self.lock2.acquire()
        self.colors = self.get_colors()
        self.drawing = Drawing(self.colors)
        self.network, self.class_names, _ = darknet.load_network(
                args.config_file,
                args.data_file,
                args.weights,
                batch_size=1
        )
        self.net_width = darknet.network_width(self.network)
        self.net_height = darknet.network_width(self.network)
        self.run()

    def get_colors(self) -> Dict[str, tuple]:
        colors = dict()
        with open(args.classes, 'r') as f:
            for line in f:
                colors[line.split('\n')[0]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return colors

    def run(self):
        threads = list()
        threads.append(Thread(target=self.detect_persons, args=(self.network, self.net_width, self.net_height, self.class_names,
                                                                self.yuv_image_queue, self.detections_queue)))
        threads.append(Thread(target=self.detect_objects, args=(self.net_width, self.net_height,
                                                                self.original_image_queue, self.detections_queue)))
        for thread in threads:
            thread.setDaemon(True)
            thread.start()

    def socket_handler(self, host:str, port:int, frame:np.ndarray) -> List[tuple]:
        client = Socket(host, port)
        client.startClient()
        client.send(frame)
        coordinates = client.recieve_text()
        client.close()
        return coordinates

    def detect_persons(self, network, net_width, net_height, class_names, 
                        yuv_image_queue, detections_queue):
        while True:
            try:
                self.lock1.acquire()
                yuv_image = yuv_image_queue.get(timeout=1)
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(yuv_image)
                frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.cuda.resize(frame, (net_width, net_height), interpolation=cv2.INTER_LINEAR)
                img_for_detect = darknet.make_image(net_width, net_height, 3)
                darknet.copy_image_from_bytes(img_for_detect, frame_resized.download().tobytes())
                detections = darknet.detect_image(network, class_names, img_for_detect)
                detections_queue.put(detections)
                darknet.free_image(img_for_detect)
            finally:
                self.lock2.release()

    def detect_objects(self, net_width, net_height,
                        original_image_queue, detections_queue):
        objects = ['hat']
        IP_list = [('127.0.0.1', 9000)]
        while True:
            try:
                self.lock2.acquire()
                original_image = original_image_queue.get(timeout=1)
                person_detections = detections_queue.get(timeout=1)
                object_detections = list()
                h, w, _ = original_image.shape
                pool = ThreadPool(processes=mp.cpu_count())
                images, coordinates = self.crop_image(person_detections, original_image, (net_width/w, net_height/h), objects)
                for id_, (objects_image, objects_coord) in enumerate(zip(images, coordinates)):
                    send_args = list()
                    for ip, object_, in zip(IP_list, objects_image):
                        send_args.append(ip+(object_,))
                    func = partial(self.socket_handler)
                    data = pool.starmap_async(func, send_args).get()
                    for obj_id, object_name in enumerate(data):
                        for detection in object_name:
                            object_detections.append((id_,)+detection+(objects_coord[obj_id],))
                pool.close()
                pool.join()
                show_frame = self.drawing.draw_person_boxes(person_detections, original_image, (net_width/w, net_height/h))
                show_frame = self.drawing.draw_object_boxes(object_detections, show_frame)
                cv2.imwrite(f'./tmp/{str(uuid.uuid4())}.jpg', show_frame)
                # cv2.imshow('img', show_frame)
                # if cv2.waitKey(1) and 0xFF == ord('q'):
                #     os._exit(0)
            finally:
                self.lock1.release()

    def crop_image(self, detections:tuple, original_image:np.ndarray, zoom:tuple, objects) -> tuple:
        images = list()
        coordinates = list()
        h, w, _ = original_image.shape
        for _, _, bbox in detections:
            left, top, right, bottom = self.drawing.bbox2points(bbox, zoom)
            image_tmp = list()
            coord_tmp = list()
            for object in objects:
                if not object == 'hook':
                    image, coord = self.normal_size(original_image, h, w, left, top, right, bottom)
                    image_tmp.append(image)
                    coord_tmp.append(coord)
                else:
                    image, coord = self.big_size(original_image, h, w, left, top, right, bottom)
                    image_tmp.append(image)
                    coord_tmp.append(coord)
            images.append(tuple(image_tmp))
            coordinates.append(tuple(coord_tmp))
        return images, coordinates
    
    def normal_size(self, image:np.ndarray, h:int, w:int, left:int, top:int, right:int, bottom:int) -> np.ndarray:
        left = left if left >= 0 else 0
        top = top if top >= 0 else 0
        right = right if right <= w else w
        bottom = bottom if bottom <= h else h
        return image[top:bottom, left:right], (left, top)
    
    def big_size(self, image:np.ndarray, h:int, w:int, left:int, top:int, right:int, bottom:int) -> np.ndarray:
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
        return image[top:bottom, left:right], (left, top)


@dataclass
class Drawing:
    colors: Dict[str, tuple]

    def bbox2points(self, bbox:tuple, zoom:tuple) -> tuple:
        zoom_x, zoom_y = zoom
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2))/zoom_x)
        xmax = int(round(x + (w / 2))/zoom_x)
        ymin = int(round(y - (h / 2))/zoom_y)
        ymax = int(round(y + (h / 2))/zoom_y)
        return (xmin, ymin, xmax, ymax)
    
    def resize_points(self, image_coordinates:tuple, bbox_coordinates:tuple) -> tuple:
        left, top, right, bottom = bbox_coordinates
        left = left+image_coordinates[0]
        top = top+image_coordinates[1]
        right = right+image_coordinates[0]
        bottom = bottom+image_coordinates[1]
        return (left, top, right, bottom)
    
    def draw_person_boxes(self, detections:tuple, image:np.ndarray, zoom:tuple) -> np.ndarray:
        for label, confidence, bbox in detections:
            left, top, right, bottom = self.bbox2points(bbox, zoom)
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[label], 1)
            cv2.putText(image, '{} [{:.2f}]'.format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.colors[label], 2)
        return image
    
    def draw_object_boxes(self, detections:tuple, image:np.ndarray) -> np.ndarray:
        for _, label, confidence, bbox_coordinates, image_coordinates in detections:
            left, top, right, bottom = self.resize_points(image_coordinates, bbox_coordinates)
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[label], 1)
            cv2.putText(image, '{} [{:.2f}]'.format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.colors[label], 2)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--data_file', type=str, default='./data/coco.data',
                        help='path to data file')
    parser.add_argument('--config_file', type=str, default='./cfg/yolov4.cfg',
                        help='path to config file')
    parser.add_argument('--weights', type=str, default='./model/yolov4.weights',
                        help='yolo weights path')
    parser.add_argument('--thresh', type=float, default=.25,
                        help='remove detections with lower confidence')
    parser.add_argument('--classes', type=str, default='./classes.txt',
                        help='label path')
    parser.add_argument('--mode', type=str, default='stream',
                        help='video or stream mode')
    args = parser.parse_args()

    original_image_queue = Queue()
    yuv_image_queue = Queue()
    detections_queue = Queue()

    camera = [f'test{idx}.mp4' for idx in range(1, 5)]
    rtsp = RtspProxy(args.mode, camera, original_image_queue, yuv_image_queue, detections_queue)
    detection = Detection(original_image_queue, yuv_image_queue, detections_queue)
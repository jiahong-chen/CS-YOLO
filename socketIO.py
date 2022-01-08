import socket
import pickle
import logging
import numpy as np

from io import BytesIO
from dataclasses import dataclass

@dataclass
class Socket:
    host: str = '127.0.0.1'
    port: str = 5000
    def __post_init__(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.client_conn = self.client_addr = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def startServer(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logging.info('[Info] Waiting for a connection.')
        
    def startClient(self):
        try:
            self.socket.connect((self.host, self.port))
            logging.info(f'[Info] Connected to {self.host} on port {self.port}')
        except socket.error as err:
            logging.error(f'[Error] Connection to {self.host} on port {self.port} failed')

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
        except AttributeError as e:
            print(e)
        logging.info('[Info] Connection is be closed already.')

    @staticmethod
    def __pack_frame(frame):
        f = BytesIO()
        np.savez_compressed(f, frame=frame)
        
        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()
        return out

    def send(self, data):
        if isinstance(data, np.ndarray):
            out = self.__pack_frame(data)
        else:
            out = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        socket = self.socket
        if(self.client_conn):
            socket = self.client_conn

        try:
            socket.sendall(out)
        except BrokenPipeError:
            logging.error('[Error] Connection broken.')
            raise
        logging.info('[Info] Send frame successfully.')

    def recieve_frame(self, socket_buffer_size=1024):
        socket = self.socket
        if(self.client_conn):
            socket = self.client_conn

        length = None
        frameBuffer = bytearray()
        while True:
            data = socket.recv(socket_buffer_size)
            frameBuffer += data
            if len(frameBuffer) == length:
                break
            while True:
                if length is None:
                    if b':' not in frameBuffer:
                        break
                    length_str, _, frameBuffer = frameBuffer.partition(b':')
                    length = int(length_str)
                if len(frameBuffer) < length:
                    break
                frameBuffer = frameBuffer[length:]
                length = None
                break
        frame = np.load(BytesIO(frameBuffer))['frame']
        logging.info('[Info] Receive frame successfully.')
        return frame
    
    def recieve_text(self, socket_buffer_size=1024):
        socket = self.socket
        if(self.client_conn):
            socket = self.client_conn
        logging.info('[Info] Receive frame successfully.')
        return pickle.loads(socket.recv(socket_buffer_size))
import socketio
import cv2
import numpy as np
from handy_concrete import handy

class HandySocket(socketio.ClientNamespace):

    def __init__(self, namespace, network):
        super().__init__(namespace)
        self._buffer = []
        self._SEQUENCE_LENGTH = 128
        self._network = network
        self._keep_feeding = True

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, buf):
        self._buffer = buf

    @property
    def network(self):
        return self._network
    
    @property
    def sequence_length(self):
        return self._SEQUENCE_LENGTH

    @property
    def keep_feeding(self):
        return self._keep_feeding

    @keep_feeding.setter
    def keep_feeding(self, keep_feeding):
        self._keep_feeding = keep_feeding

    def on_connect(self):
        print('I am connected to websocket')

    def on_frame(self, frame):

        nparr = np.frombuffer(frame, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.buffer.append(img_np)

        if len(self.buffer) == self.sequence_length and self.keep_feeding:
            self.keep_feeding, result = self.network.predict_handwashing_time_live(self.buffer)
            self.buffer = []
            if not self.keep_feeding:
                self.emit('result', result)

    def on_reset(self):
        self.buffer = []

handy_handlerer = HandySocket('/pi-frames', handy)
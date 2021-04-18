import socketio
import cv2
import numpy as np
from base64 import b64encode, b64decode
from PIL import Image

# Create websocket client
sio = socketio.Client()
@sio.on('frame', namespace='/handy-frames')
def handle_new_frame(frame):
    nparr = np.frombuffer(frame, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@sio.event(namespace='/handy-frames')
def connect():
    print("I'm connected!")

# Connect to websocket server
url = 'http://localhost:8080'

sio.connect(url, namespaces=['/handy-frames'])
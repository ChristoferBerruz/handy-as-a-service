import socketio
from websocket_handler import handy_handlerer

# Create websocket client
sio = socketio.Client()

# Connect to websocket server
url = 'http://localhost:8080'

# Define namespace or link of comunication
pi_namespace = '/pi-frames'

sio.register_namespace(handy_handlerer)

sio.connect(url, namespaces=[pi_namespace])
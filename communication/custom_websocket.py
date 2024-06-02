from websockets.sync.client import connect


class CustomWebSocket:
    def __init__(self, url):
        self.url = url
        self.wsInstance = connect(
            self.url,
            # open_timeout=60,
        )

    def send(self, message):
        self.wsInstance.send(message)

    def disconnect(self):
        self.wsInstance.close()


# wsConnector = CustomWebSocket(url="ws://127.0.0.1:8888/ws/")

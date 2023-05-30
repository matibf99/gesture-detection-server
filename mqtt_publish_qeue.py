from time import sleep

import paho.mqtt.client as mqtt
import queue


def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker")


def on_connect_failed():
    print("Connection failure")


def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT Broker")
    # client.reconnect()


def on_publish(client, userdata, mid):
    print("Message published")


class MqttPublishQueue:
    queue: queue
    connected: bool
    mqtt_client: mqtt.Client

    def __init__(self, blocking_queue):
        self.queue = blocking_queue
        self.connected = False

        self.mqtt_client = mqtt.Client(client_id="", transport="tcp", protocol=mqtt.MQTTv311)
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_disconnect = on_disconnect
        self.mqtt_client.on_publish = on_publish
        self.mqtt_client.on_connect_fail = on_connect_failed

    def run(self):
        self.mqtt_client.connect(host="127.0.0.1", port=1883)
        self.mqtt_client.loop()

        while True:
            if self.mqtt_client.is_connected() is not True:
                continue

            message = self.queue.get()
            if message is None:
                continue

            print("sending - topic: " + message.topic + ", payload: " + message.payload)
            self.mqtt_client.publish(message.topic, message.payload, retain=True)
            self.mqtt_client.loop()

    def __connect_to_broker(self):
        print("Connecting to broker...")
        self.mqtt_client.connect("localhost", 1883)

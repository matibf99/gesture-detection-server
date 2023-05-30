from dataclasses import dataclass


@dataclass
class MqttMessage:
    topic: str
    payload: str

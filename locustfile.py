from locust import HttpUser, task, between
import random

class FraudAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        payload = {
            "features": {
                "Time": random.randint(1, 100000),
                "V1": random.uniform(-2, 2),
                "V2": random.uniform(-2, 2),
                "V3": random.uniform(-2, 2),
                "V4": random.uniform(-2, 2),
                "V5": random.uniform(-2, 2),
                "V6": random.uniform(-2, 2),
                "V7": random.uniform(-2, 2),
                "V8": random.uniform(-2, 2),
                "V9": random.uniform(-2, 2),
                "V10": random.uniform(-2, 2),
                "V11": random.uniform(-2, 2),
                "V12": random.uniform(-2, 2),
                "V13": random.uniform(-2, 2),
                "V14": random.uniform(-2, 2),
                "V15": random.uniform(-2, 2),
                "V16": random.uniform(-2, 2),
                "V17": random.uniform(-2, 2),
                "V18": random.uniform(-2, 2),
                "V19": random.uniform(-2, 2),
                "V20": random.uniform(-2, 2),
                "V21": random.uniform(-2, 2),
                "V22": random.uniform(-2, 2),
                "V23": random.uniform(-2, 2),
                "V24": random.uniform(-2, 2),
                "V25": random.uniform(-2, 2),
                "V26": random.uniform(-2, 2),
                "V27": random.uniform(-2, 2),
                "V28": random.uniform(-2, 2),
                "Amount": random.uniform(1, 500)
            }
        }

        self.client.post("/predict", json=payload)

import json
import numpy as np
from datetime import datetime


def create_npy_inputs():
    # Create the inputs for the model, and save them as .npy files
    # columns 1-5 are the features, column 6 is the label
    features = extract_features()
    np.save("npy/features.npy", features)


NUM_FEATURES = 5

def extract_features() -> np.ndarray:
    user_data_file = open("json/users.json", "r")
    user_data = json.load(user_data_file)
    user_data_file.close()

    event_data_file = open("json/events.json", "r")
    event_data = json.load(event_data_file)
    event_data_file.close()

    num_events = len(event_data)

    feats = np.zeros((num_events, NUM_FEATURES + 1))

    # create event dict to speed up event lookup
    # event_dict[user_id] = [event1, event2, ...]
    # event list is sorted by timestamp
    event_dict = {}

    for event in event_data:
        user_id = event["user_id"]
        if user_id not in event_dict:
            event_dict[user_id] = []
        event_dict[user_id].append(event)

    # sort events by timestamp
    for user_id in event_dict:
        event_dict[user_id].sort(key=lambda ev: datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S"))

    # extract features
    for index, event in enumerate(event_data):
        event_class = event["code"]
        feats[index] = np.append(extract_event_features(event, event_dict), event_class)

    return feats


def extract_event_features(event_data: dict, event_dict: dict) -> np.ndarray:
    pass

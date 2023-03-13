import json
import numpy as np
import numpy.typing as npt
from datetime import datetime
from helper import *


def create_npy_inputs() -> None:
    # Create the inputs for the model, and save them as .npy files
    # columns 1-5 are the features, column 6 is the label
    features = extract_features()
    np.save("npy/features.npy", features)


NUM_FEATURES = 42


def extract_features() -> npt.NDArray[np.float64]:
    user_data_file = open("json/users.json", "r")
    user_data = json.load(user_data_file)  # maps user id (str) to preferences (list[str])
    user_data_file.close()

    event_data_file = open("json/events.json", "r")
    event_data = json.load(event_data_file)  # maps event id (str) to event (dict)
    event_data_file.close()

    num_events = len(event_data)

    feats = np.zeros((num_events, NUM_FEATURES + 1))

    # create event dict to speed up event lookup
    event_dict = {}  # maps user id (int) to list of user events (list[dict])

    for evt_id, event in event_data.items():
        user_id = event["user_id"]
        if user_id not in event_dict:
            event_dict[user_id] = []
        event_dict[user_id].append(event)

    # sort events by timestamp
    for user_id in event_dict:
        event_dict[user_id].sort(key=lambda ev: datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S"))

    # print(event_data)

    # extract features
    for index, (key, event) in enumerate(event_data.items()):
        event_class = event["code"]
        if event_class in evt_codes:
            preferences = user_data[str(event["user_id"])]
            user_event_list = event_dict[event["user_id"]]
            feats[index] = np.append(extract_event_features(event, user_event_list, preferences), get_evt_idx(event_class))

    # np.set_printoptions(threshold=np.inf)
    # print(feats[-10:, :])

    return feats


def extract_event_features(event: dict, user_event_list: list, preferences: dict) -> npt.NDArray[np.float64]:
    feats = np.zeros(NUM_FEATURES)
    # print(event)
    # print(preferences)

    user_id = event["user_id"]
    evt_ids = [evt["id"] for evt in user_event_list]
    evt_i = evt_ids.index(int(event["id"]))

    # features 0-6: email preferences
    for index, code in enumerate(pref_codes):
        feats[index] = 1 if code in preferences else 0

    # feature 7: number of previous user events, total
    # since we sorted the events by timestamp, the index of the current event is the number of previous events
    feats[7] = evt_i

    # features 8-18: number of previous user events per type
    for index, code in enumerate(evt_codes):
        feats[8 + index] = len([evt for evt in user_event_list[:evt_i] if evt["code"] == code])

    # features 19-29: number of previous user events per type, normalized by total number of previous events
    for index, code in enumerate(evt_codes):
        feats[19 + index] = feats[8 + index] / feats[7] if feats[7] > 0 else 0

    # features 30-40: number of previous user events per type for the past 10 events
    prev_10 = user_event_list[max(0, evt_i - 10):evt_i]
    for index, code in enumerate(evt_codes):
        feats[30 + index] = len([evt for evt in prev_10 if evt["code"] == code])

    # feature 41: time since first event
    first_evt_dt = datetime.strptime(user_event_list[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
    curr_evt_dt = datetime.strptime(event["timestamp"], "%Y-%m-%d %H:%M:%S")
    feats[41] = (curr_evt_dt - first_evt_dt).total_seconds()

    return feats

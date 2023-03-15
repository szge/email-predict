import json
import numpy as np
import numpy.typing as npt
from datetime import datetime
from helper import *
# from line_profiler_pycharm import profile


def create_npy_inputs() -> None:
    # Create the inputs for the model, and save them as .npy files
    # columns 1-5 are the features, column 6 is the label
    features = extract_features()
    np.save("npy/features.npy", features)


NUM_FEATURES = 43


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

    print("Creating event dict...")

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
    print("Extracting features...")
    for index, (key, event) in enumerate(event_data.items()):
        event_class = event["code"]
        user_id = event["user_id"]
        if event_class in evt_codes and str(user_id) in user_data:
            preferences = user_data[str(user_id)]
            user_event_list = event_dict[user_id]
            feats[index] = np.append(
                extract_event_features(event, user_event_list, preferences),
                get_evt_idx(event_class)
            )

    # np.set_printoptions(threshold=np.inf)
    # print(feats[-10:, :])

    return feats


# @profile
def extract_event_features(event: dict, user_event_list: list, preferences: dict) -> npt.NDArray[np.float64]:
    feats = np.zeros(NUM_FEATURES)
    # print(event)
    # print(preferences)

    evt_ids = [evt["id"] for evt in user_event_list]
    code_indices = {}
    for i, evt in enumerate(user_event_list):
        code = evt["code"]
        if code not in code_indices:
            code_indices[code] = []
        code_indices[code].append(i)

    evt_i = evt_ids.index(int(event["id"]))
    prev_evt_list = user_event_list[:evt_i]

    # features 0-7: email preferences
    for pref in preferences:
        feats[get_pref_idx(pref)] = 1

    # feature 8: number of previous user events, total
    # since we sorted the events by timestamp, the index of the current event is the number of previous events
    feats[8] = evt_i

    # features 9-19: number of previous user events per type
    for evt in prev_evt_list:
        feats[9 + get_evt_idx(evt["code"])] += 1

    # features 20-30: number of previous user events per type, normalized by total number of previous events
    for evt in prev_evt_list:
        feats[20 + get_evt_idx(evt["code"])] += 1 / feats[8] if feats[8] > 0 else 0

    # features 31-41: number of previous user events per type for the past 10 events
    prev_10 = user_event_list[max(0, evt_i - 10):evt_i]
    for evt in prev_10:
        feats[31 + get_evt_idx(evt["code"])] += 1

    # feature 42: time since first event
    # this is much faster than using datetime.strptime
    first_str = user_event_list[0]["timestamp"]
    first_evt_dt = datetime(int(first_str[:4]), int(first_str[5:7]), int(first_str[8:10]), int(first_str[11:13]),
                            int(first_str[14:16]), int(first_str[17:19]))
    curr_str = event["timestamp"]
    curr_evt_dt = datetime(int(curr_str[:4]), int(curr_str[5:7]), int(curr_str[8:10]), int(curr_str[11:13]),
                           int(curr_str[14:16]), int(curr_str[17:19]))

    feats[42] = (curr_evt_dt - first_evt_dt).total_seconds()

    return feats

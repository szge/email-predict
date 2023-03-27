import json
import numpy as np
import numpy.typing as npt
from datetime import datetime
from helper import *
# from line_profiler_pycharm import profile


def create_npy_inputs() -> None:
    # Create the inputs for the model, and save them as .npy files
    features = extract_features()
    # np.save("npy/features.npy", features)
    # get_stats()


def get_stats() -> None:
    with open("e_output/stats.txt", "w") as outf:
        features = np.load("npy/features.npy")
        codes = features[:, -1]
        outf.write("Event stats:\n")
        for (code, code_name) in zip(evt_codes, evt_names):
            outf.write(f"{code_name}: {round(sum(codes == get_evt_idx(code)) / len(codes), 3)}\n")


NUM_FEATURES = 43


user_newsletter_map = {}  # user -> newsletter ids -> events
user_ids = []


def extract_features() -> npt.NDArray[np.float64]:
    user_data_file = open("b_json/users.json", "r")
    user_data = json.load(user_data_file)  # maps user id (str) to preferences (list[str])
    user_data_file.close()

    event_data_file = open("b_json/events.json", "r")
    event_data = json.load(event_data_file)  # maps event id (str) to event (dict)
    event_data_file.close()

    newsletter_data_file = open("b_json/newsletters.json", "r")
    newsletter_data = json.load(newsletter_data_file)  # maps newsletter id (str) to newsletter (dict)
    newsletter_data_file.close()
    newsletter_ids = [int(_id) for _id in newsletter_data.keys()]
    # newsletter_ids = []

    newsletter_sent_to_user = {}  # newsletter id -> user ids

    print("Creating user-newsletter map...")
    count = 0
    for _id, newsletter in newsletter_data.items():
        # get all events with newsletter_id == _id
        _id = int(_id)
        for event in event_data.values():
            newsletter_id = int(event["newsletter_id"])
            if newsletter_id == _id:
                # get user_id
                user_id = int(event["user_id"])
                if user_id not in user_newsletter_map:
                    user_newsletter_map[user_id] = {}
                if _id not in user_newsletter_map[user_id]:
                    user_newsletter_map[user_id][_id] = []
                user_newsletter_map[user_id][_id].append(event)
                if _id not in newsletter_sent_to_user:
                    newsletter_sent_to_user[_id] = []
                newsletter_sent_to_user[_id].append(user_id)
                count += 1

    print("Sorting events by timestamp...")
    for user_id in user_newsletter_map:
        for newsletter_id in user_newsletter_map[user_id]:
            user_newsletter_map[user_id][newsletter_id].sort(
                key=lambda ev: datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S")
            )

    print("Creating features...")
    # construct feats array with size because appending is O(n*k^2)
    feats = np.zeros((count, NUM_FEATURES + 1))

    # extract features
    print("Extracting features...")
    # for index, (key, event) in enumerate(event_data.items()):
    #     event_class = event["code"]
    #     user_id = event["user_id"]
    #     if event_class in evt_codes and str(user_id) in user_data:
    #         preferences = user_data[str(user_id)]
    #         user_event_list = event_dict[user_id]
    #         feats[index] = np.append(
    #             extract_event_features(event, user_event_list, preferences),
    #             get_evt_idx(event_class)
    #         )
    #
    # # np.set_printoptions(threshold=np.inf)
    # # print(feats[-10:, :])
    #
    # return feats

    for index, (_id, newsletter) in enumerate(newsletter_data.items()):
        for user_id in newsletter_sent_to_user[int(_id)]:  # newsletter id
            preferences = user_data[str(user_id)]
            user_event_list = user_newsletter_map[user_id][int(_id)]
            feats[index] = np.append(
                extract_event_features(user_event_list, preferences),
                get_evt_idx("newsletter_sent")
            )

    return feats


def user_sent_newsletter(user_id: int, newsletter_id: int) -> bool:
    if user_id not in user_newsletter_map:
        return False
    return newsletter_id in user_newsletter_map[user_id]


def get_events(user_id: int, newsletter_id: int) -> list:
    # solve KeyError
    if not user_sent_newsletter(user_id, newsletter_id):
        return []
    return user_newsletter_map[user_id][newsletter_id]


# requires that user_newsletter_map is populated and sorted
def user_has_read_newsletter(user_id: int, newsletter_id: int) -> bool:
    for event in get_events(user_id, newsletter_id):
        if event["code"] in opened_codes:
            return True
    return False


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

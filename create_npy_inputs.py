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


NUM_FEATURES = 12

# {
#     user_id: {
#         opened: [newsletter_id1, ...],
#         unopened: [newsletter_id1, ...],
#         newsletters: {
#             newsletter_id1: [
#                 event1,
#                 ...
#             ],
#             ...
#         }
#     }
# }
user_newsletter_events = {}
user_events = {}  # maps user id (int) to list of all events (array)


def extract_features() -> npt.NDArray[np.float64]:
    user_data_file = open("b_json/users.json", "r")
    user_prefs = json.load(user_data_file)  # maps user id (str) to preferences (list[str])
    user_data_file.close()

    event_data_file = open("b_json/events.json", "r")
    event_data = json.load(event_data_file)  # maps event id (str) to event (dict)
    event_data_file.close()

    # we don't need newspaper data for this part
    # newsletter_data_file = open("b_json/newsletters.json", "r")
    # newsletter_data = json.load(newsletter_data_file)  # maps newsletter id (str) to newsletter (dict)
    # newsletter_data_file.close()

    print("Creating user-newsletter-events map...")
    for event_id, event in event_data.items():
        user_id = int(event["user_id"])
        newsletter_id = int(event["newsletter_id"])

        if user_id not in user_events:
            user_events[user_id] = []
        user_events[user_id].append(event)

        if user_id not in user_newsletter_events:
            user_newsletter_events[user_id] = {}
            user_newsletter_events[user_id]["opened"] = []
            user_newsletter_events[user_id]["unopened"] = []
            user_newsletter_events[user_id]["newsletters"] = {}
        if newsletter_id not in user_newsletter_events[user_id]["newsletters"]:
            user_newsletter_events[user_id]["newsletters"][newsletter_id] = []
        user_newsletter_events[user_id]["newsletters"][newsletter_id].append(event)

        code = event["code"]
        if code in evt_codes:
            if code in opened_codes:
                if newsletter_id in user_newsletter_events[user_id]["unopened"]:
                    user_newsletter_events[user_id]["unopened"].remove(newsletter_id)
                user_newsletter_events[user_id]["opened"].append(newsletter_id)
            else:
                user_newsletter_events[user_id]["unopened"].append(newsletter_id)

    print("Sorting events by timestamp...")
    for user_id, user_info in user_newsletter_events.items():
        for newsletter_id in user_info["newsletters"]:
            user_newsletter_events[user_id]["newsletters"][newsletter_id].sort(
                key=lambda ev: datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S")
            )
        user_info["opened"].sort()
        user_info["unopened"].sort()

    for user_id in user_events:
        user_events[user_id].sort(
            key=lambda ev: datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S")
        )

    print("Creating features...")

    # construct feats array with size because appending is O(n*k^2)
    # above as one-liner
    count = sum([len(user_info["newsletters"]) for user_id, user_info in user_newsletter_events.items()])

    feats = np.zeros((count, NUM_FEATURES + 1))

    # extract features
    count = 0
    print("Extracting features...")
    for user_id, user_info in user_newsletter_events.items():
        for newsletter_id in user_info["newsletters"]:
            preferences = user_prefs[str(user_id)]
            feats[count] = np.append(
                extract_event_features(user_id, user_info, newsletter_id, preferences),
                user_has_read_newsletter(user_id, newsletter_id)
            )
            count += 1

    # np.set_printoptions(threshold=np.inf)
    # print(feats[-10:, :])
    # np.set_printoptions(threshold=np.inf)
    # print(feats[:10, :])

    return feats


def user_sent_newsletter(user_id: int, newsletter_id: int) -> bool:
    if user_id not in user_newsletter_events:
        return False
    return newsletter_id in user_newsletter_events[user_id]["newsletters"]


def get_events(user_id: int, newsletter_id: int) -> list:
    # solve KeyError
    if not user_sent_newsletter(user_id, newsletter_id):
        return []
    return user_newsletter_events[user_id]["newsletters"][newsletter_id]


# requires that user_newsletter_map is populated and sorted
def user_has_read_newsletter(user_id: int, newsletter_id: int) -> bool:
    if not user_sent_newsletter(user_id, newsletter_id):
        return False
    return newsletter_id in user_newsletter_events[user_id]["opened"]


# requires that user has been sent newsletter
def get_time_user_sent_newsletter(user_id: int, newsletter_id: int) -> datetime:
    # turn into datetime
    return datetime.strptime(
        user_newsletter_events[user_id]["newsletters"][newsletter_id][0]["timestamp"],
        "%Y-%m-%d %H:%M:%S"
    )


def extract_event_features(
        user_id: int,
        user_info: dict,
        newsletter_id: int,
        preferences: dict
) -> npt.NDArray[np.float64]:
    feats = np.zeros(NUM_FEATURES)

    # features 0-7: email preferences
    for pref in preferences:
        if pref in evt_codes:
            feats[get_evt_idx(pref)] = 1

    # features 8: number of previous newsletters sent
    # feats[8] = len(user_newsletters[user_id]["opened"]) + len(user_newsletters[user_id]["unopened"])
    feats[8] = len(user_info["newsletters"])

    # feature 9: number of previous newsletters opened
    feats[9] = len([x for x in user_info["opened"] if x < newsletter_id])

    # feature 10: percentage of previous newsletters opened
    feats[10] = feats[9] / feats[8]

    # feature 11: time since first newsletter sent
    if user_id in user_events:
        first_newsletter_time = datetime.strptime(user_events[user_id][0]["timestamp"], "%Y-%m-%d %H:%M:%S")
        current_newsletter_time = get_time_user_sent_newsletter(user_id, newsletter_id)
        feats[11] = (current_newsletter_time - first_newsletter_time).seconds
        if feats[11] < 0:
            print(first_newsletter_time, current_newsletter_time)
    else:
        feats[11] = 0

    # BERT features

    return feats

import csv
import json
from enum import Enum


class Mode(Enum):
    FULL = "full"
    PARTIAL = "partial"


def preproc(mode: Mode = Mode.FULL):
    print("Preprocessing...")
    preproc_events(mode)
    preproc_users()


def preproc_events(mode: Mode = Mode.FULL):
    print("Preprocessing events...")
    file = open("input/emailevents_export.csv", "r")
    csv_reader = csv.reader(file)
    events_json = []
    for event in csv_reader:
        (index, code, _, user_id, event_id, timestamp) = event
        event_json = {
            "index": int(index),
            "code": code,
            "user_id": int(user_id),
            "timestamp": timestamp
        }
        events_json.append(event_json)
    file.close()

    fout = open("output/events.json", 'w')
    fout.write(json.dumps(events_json))


def preproc_users():
    print("Preprocessing users...")
    file = open("input/users_export.csv", "r")
    csv_reader = csv.reader(file)
    users_json = []
    for user in csv_reader:
        (user_id, preferences) = user
        preferences = list(preferences)
        user_json = {
            "user_id": int(user_id),
            "preferences": preferences
        }
        users_json.append(user_json)
    file.close()

    fout = open('output/users.json', 'w')
    fout.write(json.dumps(users_json))

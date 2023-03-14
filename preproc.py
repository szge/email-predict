import csv
import json
from enum import Enum
from helper import *


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
    events_json = {}
    for idx, event in enumerate(csv_reader):
        (id, code, _, user_id, event_id, timestamp) = event
        if code in evt_codes:
            event_json = {
                "id": int(id),
                "code": code,
                "user_id": int(user_id),
                "timestamp": timestamp
            }
            events_json[int(id)] = event_json
        if (mode == Mode.PARTIAL) and (idx >= 500000):
            break
    file.close()

    fout = open("json/events.json", 'w')
    fout.write(json.dumps(events_json))
    print("Output events.json")


def preproc_users():
    print("Preprocessing users...")
    file = open("input/users_export.csv", "r")
    csv_reader = csv.reader(file)
    users_json = {}
    for user in csv_reader:
        (user_id, preferences) = user
        preferences = list(preferences)
        users_json[int(user_id)] = preferences
    file.close()

    fout = open('json/users.json', 'w')
    fout.write(json.dumps(users_json))
    print("Output users.json")

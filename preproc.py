import csv
import json
from enum import Enum
from helper import *
from bs4 import BeautifulSoup


class Mode(Enum):
    FULL = "full"
    PARTIAL = "partial"


def preproc(mode: Mode = Mode.FULL):
    print("Preprocessing...")
    preproc_events(mode)
    # preproc_users()
    preproc_newsletters()


def preproc_events(mode: Mode = Mode.FULL):
    print("Preprocessing events...")
    file = open("input/emailevents_export.csv", "r")
    csv_reader = csv.reader(file)
    events_json = {}
    for idx, event in enumerate(csv_reader):
        (_id, code, newsletter_id, user_id, event_id, timestamp) = event
        if code in evt_codes:
            event_json = {
                "id": int(_id),
                "newsletter_id": newsletter_id,
                "code": code,
                "user_id": int(user_id),
                "timestamp": timestamp
            }
            events_json[int(_id)] = event_json
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


def preproc_newsletters():
    print("Preprocessing newsletters...")
    file = open("input/newsletters.csv", "r", encoding="utf-8")
    csv_reader = csv.reader(file)
    newsletters_json = {}
    next(csv_reader, None)  # skip the headers
    for newsletter in csv_reader:
        try:
            (_id, pubdate, title, _type, encoding, html, user_id, timestamp) = newsletter
            headlines_str = ""
            soup = BeautifulSoup(html, "html.parser")
            for h2 in soup.find_all("h2"):
                # print(h2.text)
                headlines_str += h2.text + ". "
            if len(title) > 0 and len(headlines_str) > 0:
                newsletters_json[int(_id)] = {
                    "title": title,
                    "headlines": headlines_str
                }
        except Exception as e:
            print(e)

    fout = open("json/newsletters.json", "w")
    fout.write(json.dumps(newsletters_json))
    print("Output newsletters.json")

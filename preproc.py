import csv
import json
from enum import Enum
from helper import *
from bs4 import BeautifulSoup


def preproc():
    print("Preprocessing...")

    # every event must refer to an existing newsletter and user
    # every user must have at least one event
    # every newsletter must have at least one event

    bad_user_list = []
    file = open("a_input/users_export.csv", "r")
    csv_reader = csv.reader(file)
    for user in csv_reader:
        (user_id, _) = user
        bad_user_list.append(int(user_id))
    file.close()

    bad_newsletter_list = []
    file = open("a_input/newsletters.csv", "r", encoding="utf-8")
    csv_reader = csv.reader(file)
    next(csv_reader, None)  # skip the headers
    for newsletter in csv_reader:
        try:
            (_id, pubdate, title, _type, encoding, html, user_id, timestamp) = newsletter
            bad_newsletter_list.append(int(_id))
        except Exception as e:
            print(e)

    file.close()

    print("Preprocessing events...")
    file = open("a_input/emailevents_export.csv", "r")
    csv_reader = csv.reader(file)
    events_json = {}
    for idx, event in enumerate(csv_reader):
        (_id, code, newsletter_id, user_id, event_id, timestamp) = event
        user_id = int(user_id)
        newsletter_id = int(newsletter_id)
        if code in evt_codes and newsletter_id in bad_newsletter_list and user_id in bad_user_list:
            event_json = {
                "newsletter_id": newsletter_id,
                "code": code,
                "user_id": user_id,
                "timestamp": timestamp
            }
            events_json[int(_id)] = event_json
    file.close()

    fout = open("b_json/events.json", 'w')
    fout.write(json.dumps(events_json))
    print("Output events.json")

    print("Preprocessing users...")
    good_user_list = set([event["user_id"] for _id, event in events_json.items()])
    file = open("a_input/users_export.csv", "r")
    csv_reader = csv.reader(file)
    users_json = {}
    for user in csv_reader:
        (user_id, prefs) = user
        user_id = int(user_id)
        if user_id in good_user_list:
            prefs = list(prefs)
            users_json[user_id] = prefs
    file.close()

    fout = open("b_json/users.json", 'w')
    fout.write(json.dumps(users_json))
    print("Output users.json")

    print("Preprocessing newsletters...")
    good_newsletters = set([event["newsletter_id"] for _id, event in events_json.items()])
    file = open("a_input/newsletters.csv", "r", encoding="utf-8")
    csv_reader = csv.reader(file)
    newsletters_json = {}
    next(csv_reader, None)  # skip the headers
    for newsletter in csv_reader:
        try:
            (_id, pubdate, title, _type, encoding, html, user_id, timestamp) = newsletter
            _id = int(_id)
            if _id in good_newsletters:
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

    fout = open("b_json/newsletters.json", "w")
    fout.write(json.dumps(newsletters_json))
    print("Output newsletters.json")

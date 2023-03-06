# this helps map between the codes and explanations for events in emailevents_export.csv
evt_codes = ["S", "B", "H", "d", "D", "L", "F", "O", "C", "U", "A"]
evt_names = ["Sent", "Soft bounce", "Hard bounce", "Deferred", "Delivered", "Loaded by proxy", "First opening",
             "Opened", "Clicked", "Unsubscribed", "Abuse complaint"]
evt_code_to_idx = {code: idx for idx, code in enumerate(evt_codes)}
def get_evt_idx(code: str) -> int: return evt_code_to_idx[code]
def get_evt_code_from_idx(idx: int) -> str: return evt_codes[idx]
def get_evt_name_from_idx(idx: int) -> str: return evt_names[idx]
def get_evt_name_from_code(code: str) -> str: return evt_names[get_evt_idx(code)]


# this helps map between the codes and explanations for preferences in users_export.csv
pref_codes = ["B", "U", "A", "D", "W", "N", "V"]
pref_names = ["Bounced", "Unsubscribed", "Active", "Subscribed to daily newsletter", "Subscribed to weekly newsletter",
              "Not sure, most likely not active", "Not sure, not being used atm"]
pref_code_to_idx = {code: idx for idx, code in enumerate(pref_codes)}
def get_pref_idx(code: str) -> int: return pref_code_to_idx[code]
def get_pref_code_from_idx(idx: int) -> str: return pref_codes[idx]
def get_pref_name_from_idx(idx: int) -> str: return pref_names[idx]
def get_pref_name_from_code(code: str) -> str: return pref_names[get_pref_idx(code)]

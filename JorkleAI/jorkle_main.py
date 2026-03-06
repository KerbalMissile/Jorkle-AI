import json
import os
import re
import random

try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

KNOWLEDGE_FILES = ["knowledge.json", "knowledge2.json"]
LEARN_PREFIX = "learn this "
STOP_WORDS = {"what", "who", "is", "where", "when", "how", "why", ""}

def speak(text):
    if not tts_engine:
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception:
        pass

def normalize(text):
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if t.startswith(("the ", "a ", "an ")):
        t = re.sub(r"^(the |a |an )", "", t).strip()
    return t

def load_knowledge():
    knowledge = {}
    for fname in KNOWLEDGE_FILES:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue
        for k, v in raw.items():
            nk = normalize(k)
            if not nk or nk in STOP_WORDS:
                continue
            if nk in knowledge:
                existing = knowledge[nk]
                if isinstance(existing, list):
                    if isinstance(v, list):
                        for item in v:
                            if item not in existing:
                                existing.append(item)
                    else:
                        if v not in existing:
                            existing.append(v)
                    knowledge[nk] = existing
                else:
                    if existing != v:
                        if isinstance(v, list):
                            combined = [existing] + [x for x in v if x != existing]
                        else:
                            combined = [existing, v]
                        knowledge[nk] = combined
            else:
                knowledge[nk] = v
    return knowledge

def save_knowledge(knowledge):
    for fname in KNOWLEDGE_FILES:
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(knowledge, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

def add_fact(text):
    t = (text or "").strip()
    if not t.lower().startswith(LEARN_PREFIX):
        return False
    remainder = t[len(LEARN_PREFIX):].strip()
    m = re.search(r"(.+?)\s+is\s+(.+)", remainder, re.IGNORECASE)
    if not m:
        m = re.search(r"(.+?)\s*[:\-]\s*(.+)", remainder)
    if not m:
        return False
    raw_key = m.group(1).strip()
    value = m.group(2).strip()
    key = normalize(raw_key)
    if not key or key in STOP_WORDS:
        return False
    knowledge = load_knowledge()
    existing = knowledge.get(key)
    if existing:
        if isinstance(existing, list):
            if value not in existing:
                existing.append(value)
                knowledge[key] = existing
        else:
            if value != existing:
                knowledge[key] = [existing, value]
    else:
        knowledge[key] = value
    save_knowledge(knowledge)
    return True

def get_answer(question):
    key_raw = question.lower().strip().rstrip("?.!")
    prefixes = [
        "what is ", "who is ", "where is ", "when is ",
        "how is ", "how do ", "how many ", "how much ", "why is "
    ]
    for p in prefixes:
        if key_raw.startswith(p):
            key_raw = key_raw[len(p):].strip()
            break
    key = normalize(key_raw)
    if not key or key in STOP_WORDS:
        return None
    knowledge = load_knowledge()
    if key in knowledge:
        v = knowledge[key]
        return random.choice(v) if isinstance(v, list) else v
    # simple semantic match
    keys = list(knowledge.keys())
    matches = [k for k in keys if key in k or k in key]
    if matches:
        v = knowledge[matches[0]]
        return random.choice(v) if isinstance(v, list) else v
    return None

def query_fact(question):
    resp = get_answer(question)
    if resp:
        print(resp)
        speak(resp)
    else:
        print("I don't know that yet.")

def main():
    print("Hey! I'm Jorkle. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit"):
            print("Bye!")
            break
        if add_fact(user_input):
            print("Got it.")
            continue
        query_fact(user_input)

if __name__ == "__main__":
    main()
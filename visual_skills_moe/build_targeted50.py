"""Build a 50-case skill-heavy targeted dataset for showcasing skill impact."""
import json
import os
import re
import random

random.seed(42)

BASE = "benchmarks/analysis"
OUT = os.path.join(BASE, "targeted_skill_50.json")

# ── Load all available datasets ──────────────────────────────────────
def load(name):
    path = os.path.join(BASE, name)
    with open(path) as f:
        data = json.load(f)
    print(f"  {name}: {len(data)} cases")
    return data

print("Loading datasets...")
ocr9 = load("ocr9_wrongset.json")
counting8 = load("counting8.json")
targeted15 = load("targeted15.json")
counting4_ocr = load("counting4_ocr_cases.json")
random100 = load("random100_videomme_seed20260212.json")
# Large wrong-case set for fishing out ASR-heavy questions
wrong952 = load("skill_wrong_952_dataset.json")

# ── Dedup helper ─────────────────────────────────────────────────────
seen_ids = set()
selected = []

def add_cases(cases, label=""):
    added = 0
    for c in cases:
        cid = c["id"]
        if cid in seen_ids:
            continue
        # Verify video exists
        vp = c.get("video_path", "")
        if not os.path.isfile(vp):
            continue
        seen_ids.add(cid)
        selected.append(c)
        added += 1
    if label:
        print(f"  Added {added} from {label} (total: {len(selected)})")

# ── ASR keyword detection ────────────────────────────────────────────
ASR_PHRASES = (
    "spoken", "narrat", "introduce", "introduced", "mention",
    "according to the video", "explain", "voice", "dialogue",
    "announce", "words", "speech", "what is the name", "company name",
    "how is ", "how are ", "how does ", "how do ",
    "what causes", "what is the reason",
    "isn't mentioned", "not mentioned", "doesn't appear",
    "not appear", "does not appear", "not feature",
)
ASR_WORD_BOUNDARY = ("said", "say", "talk", "tell", "why")

def is_asr_heavy(q):
    stem = re.split(r"\n\s*A\.", q or "", maxsplit=1)[0].lower()
    if any(k in stem for k in ASR_PHRASES):
        return True
    if any(re.search(rf"\b{k}\b", stem) for k in ASR_WORD_BOUNDARY):
        return True
    return False

# ── OCR keyword detection ────────────────────────────────────────────
OCR_PHRASES = (
    "text", "written", "display", "subtitle", "caption",
    "what is the name", "neon sign", "what time", "at what time",
)
OCR_WORD_BOUNDARY = ("sign", "read", "title", "label", "plate", "price", "clock")

def is_ocr_heavy(q):
    stem = re.split(r"\n\s*A\.", q or "", maxsplit=1)[0].lower()
    if any(k in stem for k in OCR_PHRASES):
        return True
    if any(re.search(rf"\b{k}\b", stem) for k in OCR_WORD_BOUNDARY):
        return True
    return False

# ── Counting detection (non-temporal) ────────────────────────────────
def is_counting(q):
    stem = re.split(r"\n\s*A\.", q or "", maxsplit=1)[0].lower()
    if any(k in stem for k in ("how many", "total number", "count", "number of")):
        if any(k in stem for k in ("how many times", "number of times", "how long")):
            return False  # temporal
        return True
    return False

# ── Priority 1: All OCR cases (VLM-wrong, skill can help) ───────────
print("\n=== Building targeted dataset ===")
add_cases(ocr9, "ocr9_wrongset")

# ── Priority 2: Counting cases ──────────────────────────────────────
add_cases(counting8, "counting8")
add_cases(counting4_ocr, "counting4_ocr")

# ── Priority 3: OCR + Counting from targeted15 ──────────────────────
ocr_count_from_targeted = [
    c for c in targeted15
    if c.get("category") in ("OCR Problems", "Counting Problem")
    or is_ocr_heavy(c.get("question", ""))
    or is_counting(c.get("question", ""))
]
add_cases(ocr_count_from_targeted, "targeted15 (OCR/Counting)")

# ── Priority 4: ASR-heavy from targeted15 ───────────────────────────
asr_from_targeted = [
    c for c in targeted15
    if is_asr_heavy(c.get("question", ""))
]
add_cases(asr_from_targeted, "targeted15 (ASR)")

# ── Priority 5: OCR + Counting + ASR from random100 ─────────────────
skill_heavy_100 = [
    c for c in random100
    if c.get("category") in ("OCR Problems", "Counting Problem")
    or is_asr_heavy(c.get("question", ""))
    or is_ocr_heavy(c.get("question", ""))
    or is_counting(c.get("question", ""))
]
random.shuffle(skill_heavy_100)
add_cases(skill_heavy_100, "random100 (skill-heavy)")

# ── Priority 6: OCR from wrong952 ───────────────────────────────────
ocr_from_952 = [c for c in wrong952 if c.get("category") == "OCR Problems"]
random.shuffle(ocr_from_952)
add_cases(ocr_from_952, "wrong952 (OCR)")

# ── Priority 7: ASR-heavy from wrong952 ─────────────────────────────
if len(selected) < 50:
    asr_from_952 = [
        c for c in wrong952
        if is_asr_heavy(c.get("question", ""))
        and c.get("category") not in ("OCR Problems",)  # already added
    ]
    random.shuffle(asr_from_952)
    add_cases(asr_from_952[:50 - len(selected)], "wrong952 (ASR)")

# ── Priority 8: Counting from wrong952 (non-temporal) ───────────────
if len(selected) < 50:
    count_from_952 = [
        c for c in wrong952
        if c.get("category") == "Counting Problem"
        and is_counting(c.get("question", ""))
    ]
    random.shuffle(count_from_952)
    add_cases(count_from_952[:50 - len(selected)], "wrong952 (Counting)")

# ── Priority 9: Remaining from targeted15 ────────────────────────────
if len(selected) < 50:
    add_cases(targeted15, "targeted15 (remaining)")

# ── Trim to exactly 50 ──────────────────────────────────────────────
if len(selected) > 50:
    selected = selected[:50]

# ── Normalize fields (ensure video_path exists, add missing fields) ──
for c in selected:
    c.setdefault("category", "Unknown")
    c.setdefault("domain", "Unknown")
    c.setdefault("duration", "unknown")

# ── Stats ────────────────────────────────────────────────────────────
print(f"\n=== Final dataset: {len(selected)} cases ===")
cats = {}
for c in selected:
    cat = c.get("category", "?")
    cats[cat] = cats.get(cat, 0) + 1
for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {n}")

# Count skill-trigger coverage
n_ocr = sum(1 for c in selected if is_ocr_heavy(c["question"]) or c.get("category") == "OCR Problems")
n_asr = sum(1 for c in selected if is_asr_heavy(c["question"]))
n_count = sum(1 for c in selected if is_counting(c["question"]))
print("\nSkill trigger coverage:")
print(f"  OCR-triggerable: {n_ocr}")
print(f"  ASR-triggerable: {n_asr}")
print(f"  Counting (object_detect): {n_count}")
print(f"  (some cases trigger multiple skills)")

with open(OUT, "w") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT}")

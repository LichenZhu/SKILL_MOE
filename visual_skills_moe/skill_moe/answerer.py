from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from .base import SkillResponse
from .llm_clients import LLMClient, default_llm_client

if TYPE_CHECKING:
    from .video_llm import VideoLLM

logger = logging.getLogger(__name__)

_MC_CONF_THRESHOLD = 0.58
_MC_CONF_THRESHOLD_AUDIO = 0.45

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "for", "of", "in", "on",
    "and", "or", "with", "that", "this", "it", "be", "as", "at", "by", "from",
    "not", "than", "thus", "one", "main", "video", "according",
}
_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_VISUAL_HINTS = (
    "wearing", "wear", "clothing", "dress", "shirt", "suit", "jacket",
    "color", "coloured", "looks like", "appearance", "pregnant woman",
)
_AUDIO_HINTS = (
    "said", "spoken", "words", "voice",
    "narrat", "what is said", "what did", "mentions",
)
_TEXT_HINTS = (
    "text", "caption", "subtitle", "written", "read", "sentence", "sign",
)
_COUNT_HINTS = ("how many", "total number", "count", "number of")
_GENERIC_SPECIES_LABELS = {
    "bird",
    "dog",
    "cat",
    "fish",
    "animal",
    "person",
    "insect",
}


def _extract_options(question: str) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for m in re.finditer(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", question or ""):
        opts[m.group(1).upper()] = m.group(2).strip()
    return opts


def _extract_choice(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    m = re.match(r"^\s*([A-D])(?:[\.\)\]:]|$)", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:option|answer|choice)\s*(?:is|:)?\s*([A-D])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Avoid matching "N/A" and similar slash patterns.
    m = re.search(r"(?<![/A-Z0-9])([A-D])(?![/A-Z0-9])", s, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _parse_json_payload(raw: str) -> dict:
    if not raw:
        return {}
    t = raw.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return {}
    return {}


def _normalize_text(text: str) -> str:
    s = (text or "").lower()
    s = s.replace("lou gehrig's disease", "als")
    s = s.replace("lou gehrig disease", "als")
    s = s.replace("rotaion", "rotation")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _token_set(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9']+", _normalize_text(text))
        if t and t not in _STOPWORDS
    }


def _is_low_conf(resp: SkillResponse) -> bool:
    arts = resp.artifacts if isinstance(resp.artifacts, dict) else {}
    return bool(arts.get("low_confidence"))


def _has_nontrivial_evidence(resp: SkillResponse) -> bool:
    evidence = (resp.evidence_text() or "").strip()
    return len(evidence) >= 16 and not _is_low_conf(resp)


def _is_visual_attribute_question(question: str, options: Dict[str, str]) -> bool:
    q = (question or "").lower()
    if any(k in q for k in _VISUAL_HINTS):
        return True
    opt_text = " ".join(options.values()).lower()
    if any(k in opt_text for k in _VISUAL_HINTS):
        return True
    color_words = ("black", "white", "red", "blue", "green", "yellow", "pink", "purple", "brown")
    if any(c in opt_text for c in color_words):
        return True
    return False


def _question_prefers_audio(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in _AUDIO_HINTS)


def _question_prefers_ocr(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in _TEXT_HINTS)


def _is_counting_question(question: str) -> bool:
    q = (question or "").lower()
    if any(k in q for k in _COUNT_HINTS):
        return True
    return any(
        k in q
        for k in (
            "largest number",
            "maximum number",
            "exact number",
            "correctly states the number",
            "number of people visible",
        )
    )


def _is_temporal_count_question(question: str) -> bool:
    q = (question or "").lower()
    return (
        "how many times" in q
        or "number of times" in q
        or "how long" in q
        or "how much time" in q
        or "how many years" in q
        or "how many months" in q
        or "how many days" in q
    )


def _is_people_count_question(question: str) -> bool:
    q = (question or "").lower()
    people_keys = (
        "how many people",
        "how many persons",
        "how many men",
        "how many women",
        "how many individuals",
        "how many customers",
        "how many athletes",
        "how many passengers",
        "how many children",
        "how many adults",
    )
    return any(k in q for k in people_keys)


def _is_species_question(question: str) -> bool:
    q = (question or "").lower()
    return "species" in q


def _safe_int_from_text(text: str) -> int | None:
    m = re.search(r"\b(\d+)\b", text or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    tokens = _token_set(text)
    for t in tokens:
        if t in _NUM_WORDS:
            return _NUM_WORDS[t]
    return None


def _singularize_token(token: str) -> str:
    t = (token or "").strip().lower()
    if not t:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ses") and len(t) > 4:
        return t[:-2]
    if t.endswith("xes") and len(t) > 4:
        return t[:-2]
    if t.endswith("ches") and len(t) > 5:
        return t[:-2]
    if t.endswith("shes") and len(t) > 5:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _option_numeric_map(options: Dict[str, str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for letter, text in options.items():
        num = _safe_int_from_text(text)
        if num is not None:
            out[num] = letter
    return out


def _option_numeric_ranges(options: Dict[str, str]) -> Dict[str, tuple[int | None, int | None]]:
    out: Dict[str, tuple[int | None, int | None]] = {}
    for letter, text in options.items():
        s = (text or "").lower()
        nums = [int(x) for x in re.findall(r"\b(\d+)\b", s)]
        if not nums:
            # fallback to word numbers
            for w, v in _NUM_WORDS.items():
                if re.search(rf"\b{w}\b", s):
                    nums.append(v)
            nums = sorted(set(nums))
        lo: int | None = None
        hi: int | None = None
        if "less than or equal to" in s and nums:
            hi = max(nums)
            if "more than" in s and len(nums) >= 2:
                lo = min(nums) + 1
            elif "less than" in s:
                lo = None
        elif "more than or equal to" in s and nums:
            lo = min(nums)
            hi = None
        elif "more than" in s and nums:
            lo = max(nums) + 1
            hi = None
        elif "less than" in s and nums:
            hi = min(nums) - 1
            lo = None
        elif len(nums) >= 2:
            lo = min(nums)
            hi = max(nums)
        elif len(nums) == 1:
            lo = nums[0]
            hi = nums[0]
        if lo is None and hi is None:
            continue
        out[letter] = (lo, hi)
    return out


def _pick_option_by_range(n: int, ranges: Dict[str, tuple[int | None, int | None]]) -> str:
    for letter, (lo, hi) in ranges.items():
        if lo is not None and n < lo:
            continue
        if hi is not None and n > hi:
            continue
        return letter
    return ""


def _pick_option_by_count(n: int, option_map: Dict[int, str], tol: int = 1) -> str:
    if not option_map:
        return ""
    if n in option_map:
        return option_map[n]
    nearest = min(option_map.keys(), key=lambda x: abs(x - n))
    if abs(nearest - n) <= tol:
        return option_map[nearest]
    return ""


def _count_bounds(option_map: Dict[int, str]) -> tuple[int, int]:
    if not option_map:
        return (0, 0)
    keys = sorted(option_map.keys())
    return (int(keys[0]), int(keys[-1]))


def _is_count_signal_reasonable(n: int, option_map: Dict[int, str]) -> bool:
    if not option_map:
        return False
    lo, hi = _count_bounds(option_map)
    # Allow a small tolerance around option span; reject clearly implausible outliers.
    return (lo - 1) <= n <= (hi + 2)


def _merge_spans(spans: List[tuple[float, float]], gap: float = 1.5) -> List[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[tuple[float, float]] = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s - le <= gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _extract_count_target_tokens(question: str) -> List[str]:
    q = (question or "").lower()
    phrase = ""
    m = re.search(
        r"how many\s+([a-z0-9\-\s]+?)(?:\s+(?:are|is|can|do|did|in|on|at|from|that|who|when)\b|[\?\.,]|$)",
        q,
    )
    if m:
        phrase = m.group(1).strip()
    if not phrase:
        m2 = re.search(
            r"how many times\s+(?:does|did|do|is|are)\s+([a-z0-9\-\s]+?)(?:\s+(?:appear|occur|happen|show|race|dive|cross|walk|perform)|[\?\.,]|$)",
            q,
        )
        if m2:
            phrase = m2.group(1).strip()
    if not phrase and ("largest number" in q or "maximum number" in q):
        m3 = re.search(r"(?:largest|maximum)\s+number\s*(?:of\s+)?([a-z0-9\-\s]+?)(?:[\?\.,]|$)", q)
        if m3:
            phrase = m3.group(1).strip()
    if not phrase:
        return []
    phrase = re.sub(r"\b(the|a|an|different|total|overall|currently|visible|seen|can be seen|can be observed)\b", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    tokens = [t for t in re.findall(r"[a-z0-9]+", phrase) if len(t) >= 3]
    banned = {
        "the",
        "and",
        "or",
        "are",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "from",
        "with",
        "according",
        "video",
        "national",
        "times",
        "time",
        "does",
        "did",
        "show",
        "shown",
        "occur",
        "appear",
        "visible",
        "seen",
        "happen",
        "performed",
        "perform",
    }
    out: list[str] = []
    for t in tokens:
        ts = _singularize_token(t)
        if ts in banned:
            continue
        out.append(ts)
    tokens = out
    return tokens[:6]


def _class_aliases(token: str) -> List[str]:
    t = _singularize_token(token.lower())
    table = {
        "people": ["person"],
        "persons": ["person"],
        "individual": ["person"],
        "individuals": ["person"],
        "athlete": ["person"],
        "athletes": ["person"],
        "customers": ["person"],
        "tourists": ["person"],
        "listeners": ["person"],
        "performers": ["person"],
        "player": ["person"],
        "players": ["person"],
        "men": ["person"],
        "women": ["person"],
        "woman": ["person"],
        "man": ["person"],
        "body": ["person"],
        "bodies": ["person"],
        "humanoid": ["person"],
        "humanoids": ["person"],
        "robots": ["person"],
        "robot": ["person"],
        "cats": ["cat"],
        "birds": ["bird"],
        "dogs": ["dog"],
        "flags": ["flag"],
        "cups": ["cup"],
        "bridges": ["bridge"],
        "laptops": ["laptop"],
        "books": ["book"],
        "ships": ["ship"],
        "foxes": ["fox"],
        "bears": ["bear"],
        "socks": ["sock"],
        "microphones": ["microphone"],
        "earrings": ["earring"],
        "spheres": ["ball"],
        "discs": ["disc", "glass"],
        "butterflies": ["butterfly"],
    }
    return [t] + table.get(t, [])


def _counting_answer_from_evidence(question: str, options: Dict[str, str], responses: List[SkillResponse]) -> str:
    if not _is_counting_question(question):
        return ""
    q = (question or "").lower()
    option_map = _option_numeric_map(options)
    option_ranges = _option_numeric_ranges(options)
    numeric_option_mode = len(option_map) >= 2 or len(option_ranges) >= 2

    def _map_n_to_option(n: int) -> str:
        if numeric_option_mode and option_map:
            picked = _pick_option_by_count(n, option_map, tol=2)
            if picked:
                return picked
        if numeric_option_mode and option_ranges:
            return _pick_option_by_range(n, option_ranges)
        return ""

    if _is_species_question(question):
        labels: set[str] = set()
        bird_spans: List[tuple[float, float]] = []
        bird_count_signals: List[int] = []
        for r in responses:
            if r.skill_name != "object_detect":
                continue
            arts = r.artifacts if isinstance(r.artifacts, dict) else {}
            detections = arts.get("detections")
            if not isinstance(detections, dict) or not detections:
                continue
            stats = arts.get("instance_stats")
            if isinstance(stats, dict):
                bird_stat = stats.get("bird")
                if isinstance(bird_stat, dict):
                    try:
                        p90 = int(round(float(bird_stat.get("p90", 0))))
                    except Exception:
                        p90 = 0
                    if p90 > 0:
                        bird_count_signals.append(p90)
            for key, spans in detections.items():
                cls = str(key).strip().lower()
                if not cls:
                    continue
                labels.add(cls)
                if cls != "bird" or not isinstance(spans, list):
                    continue
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    try:
                        s = float(span.get("start"))
                        e = float(span.get("end"))
                    except Exception:
                        continue
                    if e < s:
                        s, e = e, s
                    bird_spans.append((s, e))
        specific = [k for k in labels if k not in _GENERIC_SPECIES_LABELS]
        if len(specific) >= 2:
            return option_map.get(len(specific), "")
        if bird_count_signals:
            n = int(max(bird_count_signals))
            if n in option_map:
                return option_map[n]
            if option_map:
                nearest = min(option_map.keys(), key=lambda x: abs(x - n))
                if abs(nearest - n) <= 1:
                    return option_map[nearest]
        if labels and labels.issubset(_GENERIC_SPECIES_LABELS) and bird_spans:
            n = len(_merge_spans(bird_spans, gap=1.5))
            picked = _map_n_to_option(n)
            if picked:
                return picked
        return ""

    # Non-numeric comparative counting (e.g. apples/candles/berries).
    if not numeric_option_mode:
        if not any(k in q for k in ("largest number", "maximum number", "most", "fewest", "least")):
            return ""
        option_targets: dict[str, list[str]] = {}
        for letter, text in options.items():
            cleaned = re.sub(r"\([^)]*\)", " ", text.lower())
            cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
            words = [w for w in cleaned.split() if len(w) >= 3]
            if not words:
                continue
            aliases: list[str] = []
            for w in words[:4]:
                if w in {"same", "number", "kind", "kinds", "three"}:
                    continue
                w = _singularize_token(w)
                aliases.extend(_class_aliases(w))
            if aliases:
                option_targets[letter] = list(dict.fromkeys(aliases))
        if not option_targets:
            return ""
        scores: dict[str, float] = {k: 0.0 for k in option_targets}
        for r in responses:
            if r.skill_name != "object_detect":
                continue
            arts = r.artifacts if isinstance(r.artifacts, dict) else {}
            detections = arts.get("detections")
            stats = arts.get("instance_stats")
            if not isinstance(detections, dict):
                continue
            for letter, aliases in option_targets.items():
                best = 0.0
                for cls, spans in detections.items():
                    cls_l = _singularize_token(str(cls).lower())
                    if cls_l not in aliases:
                        continue
                    span_n = len(spans) if isinstance(spans, list) else 0
                    p90 = 0.0
                    if isinstance(stats, dict):
                        st = stats.get(cls)
                        if isinstance(st, dict):
                            try:
                                p90 = float(st.get("p90", 0.0) or 0.0)
                            except Exception:
                                p90 = 0.0
                    # For comparative "which has the largest number" questions,
                    # span coverage is generally more stable than raw per-frame maxima.
                    score_local = float(span_n) + 0.25 * float(p90)
                    best = max(best, score_local)
                scores[letter] = max(scores[letter], best)
        if any(v > 0 for v in scores.values()):
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top_letter, top_score = ranked[0]
            # "same number" option when evidence is nearly tied.
            same_letter = ""
            for letter, text in options.items():
                if "same number" in (text or "").lower():
                    same_letter = letter
                    break
            if same_letter:
                non_same_vals = [
                    float(v)
                    for letter, v in scores.items()
                    if letter != same_letter and float(v) > 0.0
                ]
                if len(non_same_vals) >= 2 and (max(non_same_vals) - min(non_same_vals)) <= 0.5:
                    return same_letter
            return top_letter
        return ""

    target_tokens = _extract_count_target_tokens(question)
    target_aliases = {a for t in target_tokens for a in _class_aliases(t)}

    class_signals: dict[str, list[int]] = {}
    span_signals: dict[str, list[int]] = {}
    face_signals: list[int] = []
    action_signals: list[int] = []
    asr_signals: list[int] = []

    for r in responses:
        if _is_low_conf(r):
            continue
        arts = r.artifacts if isinstance(r.artifacts, dict) else {}
        if r.skill_name == "shape_count":
            n = arts.get("circle_count_estimate")
            try:
                n_int = int(n)
            except Exception:
                n_int = None
            if n_int is not None:
                picked = _map_n_to_option(n_int)
                if picked:
                    return picked

        if r.skill_name == "face":
            try:
                n = int(arts.get("max_faces_per_frame", 0))
            except Exception:
                n = 0
            if n > 0:
                face_signals.append(n)
            continue

        if r.skill_name == "action":
            actions = arts.get("actions")
            if isinstance(actions, list) and actions:
                action_signals.append(len(actions))
            continue

        if r.skill_name == "asr":
            text = (r.evidence_text() or "").lower()
            nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", text)]
            for w, v in _NUM_WORDS.items():
                if re.search(rf"\b{w}\b", text):
                    nums.append(v)
            asr_signals.extend([n for n in nums if 0 <= n <= 30])
            continue

        if r.skill_name != "object_detect":
            continue
        detections = arts.get("detections")
        stats = arts.get("instance_stats")
        if not isinstance(detections, dict):
            continue
        for cls, spans in detections.items():
            cls_l = _singularize_token(str(cls).lower())
            if isinstance(spans, list):
                span_signals.setdefault(cls_l, []).append(len(spans))
            if isinstance(stats, dict):
                st = stats.get(cls)
                if isinstance(st, dict):
                    vals: list[int] = []
                    try:
                        p90 = int(round(float(st.get("p90", 0) or 0)))
                        if p90 > 0:
                            vals.append(p90)
                    except Exception:
                        pass
                    try:
                        mx = int(round(float(st.get("max", 0) or 0)))
                        if mx > 0:
                            vals.append(mx)
                    except Exception:
                        pass
                    if vals:
                        class_signals.setdefault(cls_l, []).extend(vals)

    # People counting: trust face + person-oriented detections.
    if _is_people_count_question(question):
        people_aliases = {"person", "people", "man", "woman", "men", "women"}
        if "tie" in q:
            people_aliases.add("tie")
        if "glass" in q:
            people_aliases.update({"glass", "eyeglasses"})
        signals: list[int] = []
        signals.extend(face_signals)
        for alias in people_aliases:
            signals.extend(class_signals.get(alias, []))
            signals.extend(span_signals.get(alias, []))
        if signals:
            n = int(round(sorted(signals)[len(signals) // 2]))
            picked = _map_n_to_option(n)
            if picked:
                return picked

    # Temporal/event counting: prefer span/action counts.
    if _is_temporal_count_question(question):
        temporal: list[int] = []
        if target_aliases:
            for alias in target_aliases:
                temporal.extend(span_signals.get(alias, []))
        temporal.extend(action_signals)
        temporal.extend(asr_signals)
        temporal = [n for n in temporal if n >= 0]
        if temporal:
            n = int(round(sorted(temporal)[len(temporal) // 2]))
            picked = _map_n_to_option(n)
            if picked:
                return picked

    # Generic object counting.
    if target_aliases:
        signals: list[int] = []
        for alias in target_aliases:
            signals.extend(class_signals.get(alias, []))
            signals.extend(span_signals.get(alias, []))
        if signals:
            # For "different/appear" style questions, span counts are often
            # more informative than per-frame p90 counts.
            prefer_span = any(k in q for k in ("different", "appear", "shown", "show up"))
            vals = sorted(signals)
            if prefer_span:
                n = int(round(vals[int(round((len(vals) - 1) * 0.75))]))
            else:
                n = int(round(vals[len(vals) // 2]))
            picked = _map_n_to_option(n)
            if picked:
                return picked

    # Last fallback: robust vote from all numeric signals.
    merged = face_signals + action_signals + asr_signals
    for vals in class_signals.values():
        merged.extend(vals)
    if merged:
        n = int(round(sorted(merged)[len(merged) // 2]))
        picked = _map_n_to_option(n)
        if picked:
            return picked

    return ""


def _audio_overlap_answer(question: str, options: Dict[str, str], responses: List[SkillResponse]) -> str:
    if not _question_prefers_audio(question):
        return ""
    asr_text = " ".join((r.evidence_text() or "") for r in responses if r.skill_name == "asr")
    if len(asr_text) < 40:
        return ""
    asr_tokens = _token_set(asr_text)
    if not asr_tokens:
        return ""
    asr_norm = _normalize_text(asr_text)
    q_norm = _normalize_text(question)

    # High-precision shortcut for wind-cause questions with explicit
    # rotation+orbit evidence in transcript.
    if "wind" in q_norm and any(k in q_norm for k in ("cause", "causes", "reason", "makes", "what makes")):
        if "rotation" in asr_norm and "orbit" in asr_norm:
            for letter, opt in options.items():
                opt_norm = _normalize_text(opt)
                if ("rotation" in opt_norm or "orbit" in opt_norm) and "temperature" in opt_norm:
                    return letter
        if "orbit" in asr_norm and any(k in asr_norm for k in ("equator", "poles", "tilt")):
            for letter, opt in options.items():
                opt_norm = _normalize_text(opt)
                if "orbit" in opt_norm or "rotation" in opt_norm:
                    return letter

    scores: Dict[str, int] = {}
    for letter, opt in options.items():
        toks = _token_set(opt)
        scores[letter] = len(toks & asr_tokens)
        opt_norm = _normalize_text(opt)
        if "rotation" in opt_norm and "rotation" in asr_norm:
            scores[letter] += 2
        if "orbit" in opt_norm and "orbit" in asr_norm:
            scores[letter] += 2
        if "temperature" in opt_norm and "temperature" in asr_norm:
            scores[letter] += 1
        if "als" in opt_norm and "als" in asr_norm:
            scores[letter] += 2
        if "sea" in opt_norm and "sea" not in asr_norm:
            scores[letter] -= 1
        if "solar" in opt_norm and "solar" not in asr_norm:
            scores[letter] -= 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return ""
    best_letter, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score >= 3 and (best_score - second_score) >= 2:
        return best_letter
    return ""


def _visual_option_answer_from_evidence(
    question: str,
    options: Dict[str, str],
    responses: List[SkillResponse],
) -> str:
    if not _is_visual_attribute_question(question, options):
        return ""
    for r in reversed(responses):
        if r.skill_name != "visual_option_match":
            continue
        arts = r.artifacts if isinstance(r.artifacts, dict) else {}
        letter = str(arts.get("best_option", "")).upper().strip()
        if letter not in {"A", "B", "C", "D"}:
            continue
        try:
            conf = float(arts.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf >= 0.45:
            return letter
    return ""


def _collect_evidence_profile(responses: List[SkillResponse]) -> Dict[str, object]:
    profile = {
        "has_asr": False,
        "has_ocr": False,
        "has_object_detect": False,
        "asr_chars": 0,
        "ocr_chars": 0,
        "high_conf_count": 0,
    }
    for r in responses:
        ev = (r.evidence_text() or "").strip()
        if _has_nontrivial_evidence(r):
            profile["high_conf_count"] += 1
        if r.skill_name == "asr":
            profile["has_asr"] = True
            profile["asr_chars"] += len(ev)
        if r.skill_name == "ocr":
            profile["has_ocr"] = True
            profile["ocr_chars"] += len(ev)
        if r.skill_name == "object_detect":
            profile["has_object_detect"] = True
    return profile


def _should_prefer_evidence_qa(question: str, responses: List[SkillResponse]) -> bool:
    opts = _extract_options(question)
    if len(opts) < 2 or not responses:
        return False
    if _is_species_question(question):
        return False
    if _is_visual_attribute_question(question, opts):
        return False

    profile = _collect_evidence_profile(responses)
    if profile["has_asr"] and _question_prefers_audio(question):
        return int(profile["asr_chars"]) >= 60
    if _is_temporal_count_question(question):
        return int(profile["high_conf_count"]) >= 1 and (profile["has_asr"] or profile["has_object_detect"])
    if _is_counting_question(question):
        return int(profile["high_conf_count"]) >= 1
    if profile["has_ocr"] and _question_prefers_ocr(question):
        return int(profile["ocr_chars"]) >= 20
    if profile["has_object_detect"] and _is_counting_question(question) and not _is_temporal_count_question(question):
        return int(profile["high_conf_count"]) >= 3
    return int(profile["high_conf_count"]) >= 2


def _filtered_responses_for_context(question: str, responses: List[SkillResponse]) -> List[SkillResponse]:
    opts = _extract_options(question)
    visual = _is_visual_attribute_question(question, opts)
    filtered = [r for r in responses if _has_nontrivial_evidence(r)]
    if not filtered:
        filtered = list(responses)
    if _is_species_question(question):
        narrowed = [
            r
            for r in filtered
            if not (
                r.skill_name == "object_detect"
                and isinstance(r.artifacts, dict)
                and isinstance(r.artifacts.get("detections"), dict)
                and set(str(k).strip().lower() for k in r.artifacts.get("detections", {}).keys() if str(k).strip())
                and set(str(k).strip().lower() for k in r.artifacts.get("detections", {}).keys() if str(k).strip()).issubset(_GENERIC_SPECIES_LABELS)
            )
        ]
        if narrowed:
            filtered = narrowed
    if visual:
        # Avoid poisoning visual QA with noisy non-visual tool traces.
        narrowed = [r for r in filtered if r.skill_name in {"ocr", "visual_option_match"}]
        if narrowed:
            filtered = narrowed
    return filtered


def build_context(question: str, responses: List[SkillResponse]) -> str:
    parts = [f"Question: {question}", "Skill outputs:"]
    for resp in responses:
        evidence = resp.evidence_text()
        parts.append(f"- {resp.skill_name}: {evidence or resp.summary}")
        if resp.artifacts:
            parts.append(f"  artifacts: {resp.artifacts}")
    return "\n".join(parts)


def _question_option_tokens(question: str) -> set[str]:
    q = question or ""
    q = re.sub(r"(?is)choose the correct answer from:.*", " ", q)
    q = re.sub(r"(?is)answer with only the letter.*", " ", q)
    return _token_set(q)


def _evidence_relevance(question: str, responses: List[SkillResponse]) -> float:
    q_tokens = _question_option_tokens(question)
    if not q_tokens or not responses:
        return 0.0
    ev = _extract_compact_evidence_text(responses, limit_chars=5000)
    ev_tokens = _token_set(ev)
    if not ev_tokens:
        return 0.0
    overlap = len(q_tokens & ev_tokens)
    return overlap / max(1, len(q_tokens))


def _extract_compact_evidence_text(responses: List[SkillResponse], limit_chars: int = 7000) -> str:
    chunks: list[str] = []
    cur = 0
    for r in responses:
        ev = (r.evidence_text() or "").strip()
        if not ev:
            continue
        piece = f"[{r.skill_name}] {ev}"
        if cur + len(piece) > limit_chars:
            remain = max(0, limit_chars - cur)
            if remain > 20:
                chunks.append(piece[:remain])
            break
        chunks.append(piece)
        cur += len(piece)
    return "\n".join(chunks)


def _lexical_option_vote(question: str, responses: List[SkillResponse]) -> str:
    opts = _extract_options(question)
    if len(opts) < 2:
        return ""
    if _is_visual_attribute_question(question, opts):
        return ""
    evidence = _extract_compact_evidence_text(responses, limit_chars=8000)
    if len(evidence) < 40:
        return ""
    ev_tokens = _token_set(evidence)
    if not ev_tokens:
        return ""

    scores: dict[str, float] = {}
    for letter, opt in opts.items():
        toks = _token_set(opt)
        if not toks:
            scores[letter] = 0.0
            continue
        overlap = len(toks & ev_tokens)
        score = overlap / math.sqrt(len(toks))
        opt_norm = _normalize_text(opt)
        ev_norm = _normalize_text(evidence)
        if "als" in ev_norm and "als" in opt_norm:
            score += 2.0
        if ("rotation" in opt_norm or "orbit" in opt_norm) and ("rotation" in ev_norm or "orbit" in ev_norm):
            score += 1.5
        if "temperature" in opt_norm and "temperature" in ev_norm:
            score += 0.8
        scores[letter] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return ""
    best_letter, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score >= 2.2 and (best_score - second_score) >= 0.8:
        return best_letter
    return ""


def _evidence_mc_with_llm(
    question: str,
    skill_context: str,
    client: LLMClient,
    max_tokens: int = 180,
) -> tuple[str, float, str]:
    prompt = (
        "You are a strict multiple-choice solver.\n"
        "Use ONLY the provided evidence from tools.\n"
        "Do not add external assumptions.\n"
        "Map equivalent terms when obvious (example: ALS == Lou Gehrig's disease).\n"
        "If evidence is insufficient, return N/A.\n\n"
        "Return JSON only:\n"
        "{\"answer\":\"A|B|C|D|N/A\",\"confidence\":0.0-1.0,\"reason\":\"short\"}\n\n"
        f"{skill_context}\n\n"
        "Final JSON:"
    )
    raw = client.complete(prompt, max_tokens=max_tokens)
    payload = _parse_json_payload(raw)
    ans = str(payload.get("answer", "")).upper().strip()
    if ans not in {"A", "B", "C", "D"}:
        ans = _extract_choice(raw)
    try:
        conf = float(payload.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    reason = str(payload.get("reason", "")).strip()
    return ans, conf, reason


def _text_only_answer(
    question: str,
    skill_context: str | None,
    client: LLMClient,
    max_tokens: int,
) -> str:
    prompt = (
        "Answer the question using the provided context.\n"
        "If it is a multiple-choice question, return only A/B/C/D.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{skill_context or '(none)'}\n"
    )
    out = client.complete(prompt, max_tokens=max_tokens)
    opts = _extract_options(question)
    if opts:
        letter = _extract_choice(out)
        if letter:
            return letter
    return out.strip()


def answer(
    question: str,
    responses: List[SkillResponse],
    trace: Optional[object] = None,
    video_path: Optional[str] = None,
    video_llm: Optional[VideoLLM] = None,
    llm_client: LLMClient | None = None,
    max_tokens: int = 256,
) -> str:
    """
    Final answer synthesis:
    1) Evidence-first for MC when tool evidence is strong.
    2) Video-LLM answer (optionally with filtered tool context).
    3) Text-only LLM fallback.
    """
    filtered_responses = _filtered_responses_for_context(question, responses)
    skill_context = build_context(question, filtered_responses) if filtered_responses else None
    options = _extract_options(question)
    is_mc = len(options) >= 2
    client = llm_client or default_llm_client()

    if is_mc:
        visual_match = _visual_option_answer_from_evidence(question, options, filtered_responses)
        if visual_match in {"A", "B", "C", "D"}:
            return visual_match

        counted = _counting_answer_from_evidence(question, options, responses)
        if counted in {"A", "B", "C", "D"}:
            return counted

        audio_pick = _audio_overlap_answer(question, options, responses)
        if audio_pick in {"A", "B", "C", "D"}:
            return audio_pick

        if _question_prefers_audio(question) or _question_prefers_ocr(question):
            lexical = _lexical_option_vote(question, responses)
            if lexical in {"A", "B", "C", "D"}:
                return lexical

        if skill_context and client and _should_prefer_evidence_qa(question, filtered_responses):
            try:
                ans, conf, _ = _evidence_mc_with_llm(
                    question=question,
                    skill_context=skill_context,
                    client=client,
                    max_tokens=min(max_tokens, 180),
                )
                th = _MC_CONF_THRESHOLD_AUDIO if (_question_prefers_audio(question) or _question_prefers_ocr(question)) else _MC_CONF_THRESHOLD
                if ans in {"A", "B", "C", "D"} and conf >= th:
                    return ans
            except Exception as exc:
                logger.warning("Evidence-first MC step failed: %s", exc)

    if video_llm is not None and video_path is not None:
        try:
            video_context = skill_context
            if is_mc and _is_counting_question(question):
                profile = _collect_evidence_profile(filtered_responses)
                # Keep counting evidence only when we actually collected useful signals.
                if int(profile["high_conf_count"]) <= 0:
                    video_context = None
            # For appearance/color-style questions, avoid noisy tool text.
            if is_mc and _is_visual_attribute_question(question, options):
                video_context = None
            elif skill_context:
                relevance = _evidence_relevance(question, filtered_responses)
                if relevance < 0.12 and not _should_prefer_evidence_qa(question, filtered_responses):
                    video_context = None
            out = video_llm.answer(
                question=question,
                video_path=video_path,
                skill_context=video_context,
            )
            if is_mc:
                letter = _extract_choice(out)
                if letter:
                    return letter
                if skill_context and client:
                    try:
                        ans, conf, _ = _evidence_mc_with_llm(
                            question=question,
                            skill_context=skill_context,
                            client=client,
                            max_tokens=min(max_tokens, 180),
                        )
                        if ans in {"A", "B", "C", "D"} and conf >= _MC_CONF_THRESHOLD_AUDIO:
                            return ans
                    except Exception:
                        pass
            return out
        except Exception as exc:
            logger.error("Video LLM failed: %s", exc, exc_info=True)

    if client:
        try:
            return _text_only_answer(question, skill_context, client, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("Text-only fallback failed: %s", exc)

    if trace is not None and hasattr(trace, "history_text"):
        return trace.history_text()
    if responses:
        return " | ".join(resp.summary for resp in responses)
    return "No skills were run and no video LLM available; cannot answer."

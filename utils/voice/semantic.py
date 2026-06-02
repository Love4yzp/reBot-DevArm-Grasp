"""Rule-based semantic parsing for short robot voice commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
from typing import Any


_RICH_TAG_RE = re.compile(r"<\|[^|]+?\|>")
_SPACE_RE = re.compile(r"\s+")

_ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "grasp": (
        "抓",
        "拿",
        "夹",
        "取",
        "捡",
        "抓取",
        "拾取",
        "想要",
        "我要",
        "需要",
        "递给我",
        "给我拿",
        "pick up",
        "pick",
        "pickup",
        "grab",
        "take",
        "want",
        "need",
    ),
    "stop": ("停止", "停下", "别动", "不要动", "暂停", "stop"),
    "home": ("回零", "回到初始", "回家", "归位", "home"),
    "open_gripper": ("张开", "打开夹爪", "松开", "open"),
    "close_gripper": ("闭合", "合上", "夹紧", "close"),
}

_FILLERS = (
    "帮我",
    "请",
    "麻烦",
    "一下",
    "那个",
    "这个",
    "把",
    "给我",
    "可以",
    "帮忙",
    "机器人",
    "机械臂",
    "我",
    "的",
    "i",
    "me",
    "my",
    "a",
    "an",
    "the",
    "this",
    "that",
    "please",
    "can you",
    "could you",
)

_LOCATION_ALIASES: dict[str, str] = {
    "桌子上": "on_table",
    "桌面上": "on_table",
    "台面上": "on_table",
    "左边": "left",
    "右边": "right",
    "中间": "center",
    "中央": "center",
    "前面": "front",
    "后面": "back",
    "靠左": "left",
    "靠右": "right",
    "最左": "leftmost",
    "最右": "rightmost",
    "left": "left",
    "right": "right",
    "center": "center",
    "middle": "center",
    "front": "front",
    "back": "back",
    "on the table": "on_table",
}

_COLOR_ALIASES: dict[str, str] = {
    "浅蓝色": "light blue",
    "淡蓝色": "light blue",
    "天蓝色": "light blue",
    "蓝色": "blue",
    "红色": "red",
    "绿色": "green",
    "黄色": "yellow",
    "黑色": "black",
    "白色": "white",
    "灰色": "gray",
    "橙色": "orange",
    "紫色": "purple",
    "粉色": "pink",
    "light blue": "light blue",
    "blue": "blue",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "orange": "orange",
    "purple": "purple",
    "pink": "pink",
}

_TARGET_ALIASES: dict[str, str] = {
    "浅蓝色咖啡杯": "coffee cup",
    "咖啡杯": "coffee cup",
    "水杯": "cup",
    "杯子": "cup",
    "杯": "cup",
    "瓶装水": "water bottle",
    "矿泉水": "water bottle",
    "水瓶": "water bottle",
    "瓶子": "bottle",
    "香蕉": "banana",
    "黄香蕉": "banana",
    "工具": "tool",
    "扳手": "tool",
    "螺丝刀": "tool",
    "手机": "cell phone",
    "电话": "cell phone",
    "盒子": "box",
    "箱子": "box",
    "红色物体": "object",
    "绿色物体": "object",
    "物体": "object",
    "东西": "object",
    "coffee cup": "coffee cup",
    "cup": "cup",
    "water bottle": "water bottle",
    "bottle": "bottle",
    "banana": "banana",
    "tool": "tool",
    "cell phone": "cell phone",
    "phone": "cell phone",
    "box": "box",
    "cube": "cube",
    "block": "cube",
}

_CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass(frozen=True)
class TargetInfo:
    text_zh: str | None
    yolo_phrase: str | None
    attributes: dict[str, Any]
    confidence: float
    needs_vocab_review: bool


@dataclass(frozen=True)
class CommandIntent:
    raw_text: str
    clean_text: str
    action: str
    target: TargetInfo | None
    yolo_class_phrases: list[str]
    alternatives: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_command(text: str) -> CommandIntent:
    raw_text = text or ""
    normalized = normalize_text(raw_text)
    action = detect_action(normalized)
    attributes, reduced = extract_attributes(normalized)
    target_phrase = reduce_to_target_phrase(reduced)
    target, alternatives = resolve_target(target_phrase, attributes)
    yolo_phrases = [target.yolo_phrase] if target and target.yolo_phrase else []

    return CommandIntent(
        raw_text=raw_text,
        clean_text=target_phrase or normalized,
        action=action,
        target=target,
        yolo_class_phrases=yolo_phrases,
        alternatives=alternatives,
    )


def normalize_text(text: str) -> str:
    text = _RICH_TAG_RE.sub("", text)
    text = text.strip().lower()
    text = re.sub(r"[，。！？、；：,.!?;:\"'`~()\[\]{}]", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def detect_action(text: str) -> str:
    for action, aliases in _ACTION_ALIASES.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            return action
    return "unknown"


def extract_attributes(text: str) -> tuple[dict[str, Any], str]:
    reduced = text
    attrs: dict[str, Any] = {"color": None, "location": None, "quantity": None}

    for alias in sorted(_LOCATION_ALIASES, key=len, reverse=True):
        if _contains_alias(reduced, alias):
            attrs["location"] = _LOCATION_ALIASES[alias]
            reduced = _remove_alias(reduced, alias, count=1)
            break

    for alias in sorted(_COLOR_ALIASES, key=len, reverse=True):
        if _contains_alias(reduced, alias):
            attrs["color"] = _COLOR_ALIASES[alias]
            reduced = _remove_alias(reduced, alias, count=1)
            break

    quantity = _extract_quantity(reduced)
    if quantity is not None:
        attrs["quantity"] = quantity
        reduced = re.sub(r"第?[一二两三四五六七八九十0-9]+个", " ", reduced, count=1)

    return attrs, _SPACE_RE.sub(" ", reduced).strip()


def reduce_to_target_phrase(text: str) -> str:
    reduced = text
    for aliases in _ACTION_ALIASES.values():
        for alias in sorted(aliases, key=len, reverse=True):
            reduced = _remove_alias(reduced, alias)
    for filler in sorted(_FILLERS, key=len, reverse=True):
        reduced = _remove_alias(reduced, filler)
    return _SPACE_RE.sub(" ", reduced).strip()


def resolve_target(target_phrase: str, attributes: dict[str, Any]) -> tuple[TargetInfo | None, list[dict[str, Any]]]:
    if not target_phrase:
        return None, []

    candidates = _rank_targets(target_phrase)
    if not _contains_cjk(target_phrase):
        english_candidates = [item for item in candidates if not _contains_cjk(item[0])]
        if english_candidates:
            candidates = english_candidates
    best = candidates[0] if candidates else None
    alternatives = [
        {"text": alias, "yolo_phrase": yolo, "score": round(score, 3)}
        for alias, yolo, score in candidates[1:4]
    ]

    if best and best[2] >= 0.62:
        alias, yolo_object, score = best
        yolo_phrase = _compose_yolo_phrase(yolo_object, attributes)
        text_zh = alias if _contains_cjk(alias) else target_phrase
        return (
            TargetInfo(
                text_zh=text_zh,
                yolo_phrase=yolo_phrase,
                attributes=attributes,
                confidence=round(min(0.98, score), 3),
                needs_vocab_review=False,
            ),
            alternatives,
        )

    return (
        TargetInfo(
            text_zh=target_phrase,
            yolo_phrase=target_phrase,
            attributes=attributes,
            confidence=0.35,
            needs_vocab_review=True,
        ),
        alternatives,
    )


def _rank_targets(target_phrase: str) -> list[tuple[str, str, float]]:
    scored: list[tuple[str, str, float]] = []
    compact = target_phrase.replace(" ", "")
    for alias, yolo in _TARGET_ALIASES.items():
        alias_compact = alias.replace(" ", "")
        yolo_compact = yolo.replace(" ", "")
        if alias_compact == compact:
            score = 0.99
        elif yolo_compact == compact:
            score = 0.98
        elif _contains_alias(target_phrase, alias) or (_contains_cjk(alias) and alias_compact in compact):
            score = 0.96 if len(alias_compact) >= 2 else 0.84
        elif target_phrase in alias or compact in alias_compact:
            score = 0.88
        else:
            score = max(
                SequenceMatcher(None, compact, alias_compact).ratio(),
                SequenceMatcher(None, target_phrase, yolo).ratio(),
            )
        scored.append((alias, yolo, score))
    return sorted(
        scored,
        key=lambda item: (item[2], item[0].replace(" ", "") == compact),
        reverse=True,
    )


def _compose_yolo_phrase(yolo_object: str, attributes: dict[str, Any]) -> str:
    color = attributes.get("color")
    if color and color not in yolo_object:
        return f"{color} {yolo_object}"
    return yolo_object


def _extract_quantity(text: str) -> int | None:
    match = re.search(r"第?([一二两三四五六七八九十0-9]+)个", text)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    if token in _CHINESE_NUMBERS:
        return _CHINESE_NUMBERS[token]
    return None


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _contains_alias(text: str, alias: str) -> bool:
    return re.search(_alias_pattern(alias), text) is not None


def _remove_alias(text: str, alias: str, count: int = 0) -> str:
    return re.sub(_alias_pattern(alias), " ", text, count=count)


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias.strip()).replace(r"\ ", r"\s+")
    if re.search(r"[a-z0-9]", alias) and not _contains_cjk(alias):
        return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return escaped

"""
app.py – 画像内の個人情報を匿名化ラベルで置換する Streamlit アプリ
====================================================================
iPhone Safari 完全対応・タッチ2点指定モザイク版
・PC    → ダブルクリックで1点目、ダブルクリックで2点目
・iPhone → 長押しタップで1点目、長押しタップで2点目
・自動検出（帳票項目名ベース + 正規表現）
・FAX行の電話番号は自動除外
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import streamlit as st


# =====================================================================
# 0. 起動時チェック
# =====================================================================

def _check_dependencies() -> list[str]:
    errors: list[str] = []
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        errors.append(
            "❌ **Tesseract が見つかりません。**\n"
            "- Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-jpn`"
        )
    return errors


# =====================================================================
# 1. データクラス
# =====================================================================

@dataclass
class OCRBox:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class MaskRegion:
    original_text: str
    label: str
    anon_text: str
    left: int
    top: int
    right: int
    bottom: int
    source: str = "ner"

    @property
    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)


# =====================================================================
# 2. 匿名化ラベルマネージャー
# =====================================================================

class AnonymizationManager:

    @staticmethod
    def _person_label(index: int) -> str:
        result = ""
        i = index
        while True:
            result = chr(ord("A") + i % 26) + result
            i = i // 26 - 1
            if i < 0:
                break
        return result

    _HIRAGANA = list(
        "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        "はひふへほまみむめもやゆよらりるれろわをん"
    )
    _KATAKANA = list(
        "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
        "ハヒフヘホマミムメモヤユヨラリルレロワヲン"
    )

    @classmethod
    def _location_label(cls, index: int) -> str:
        return f"地点{cls._HIRAGANA[index]}" if index < len(cls._HIRAGANA) else f"地点{index + 1}"

    @classmethod
    def _org_label(cls, index: int) -> str:
        return f"組織{cls._KATAKANA[index]}" if index < len(cls._KATAKANA) else f"組織{index + 1}"

    @staticmethod
    def _phone_label(index: int) -> str:
        return f"000-0000-{index + 1:04d}"

    @staticmethod
    def _postal_label(index: int) -> str:
        return f"〒000-{index + 1:04d}"

    @staticmethod
    def _email_label(index: int) -> str:
        return f"anonymous{index + 1:03d}@example.com"

    @staticmethod
    def _date_label(index: int) -> str:
        return f"0000年00月{index + 1:02d}日"

    @staticmethod
    def _account_label(index: int) -> str:
        return f"口座{index + 1:07d}"

    @staticmethod
    def _mynumber_label(index: int) -> str:
        return f"0000-0000-{index + 1:04d}"

    @staticmethod
    def _other_label(index: int) -> str:
        return f"XXX-{index + 1:03d}"

    CATEGORY_MAP: dict[str, str] = {
        "Person": "person", "PERSON": "person",
        "ORG": "org", "Organization": "org",
        "LOC": "location", "Location": "location",
        "GPE": "location", "FAC": "location", "Facility": "location",
        "PRODUCT": "other", "Product": "other",
        "EVENT": "other", "Event": "other",
        "NORP": "org",
        "電話番号": "phone",
        "郵便番号": "postal",
        "メール": "email",
        "住所": "location",
        "日付(生年月日)": "date",
        "口座番号": "account",
        "マイナンバー": "mynumber",
    }

    CATEGORY_JP: dict[str, str] = {
        "person": "人名", "location": "地名", "org": "組織",
        "phone": "電話番号", "postal": "郵便番号", "email": "メール",
        "date": "日付", "account": "口座番号", "mynumber": "マイナンバー",
        "other": "その他",
    }

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, str]] = {}
        self._counters: dict[str, int] = {}
        self._generators = {
            "person":   self._person_label,
            "location": self._location_label,
            "org":      self._org_label,
            "phone":    self._phone_label,
            "postal":   self._postal_label,
            "email":    self._email_label,
            "date":     self._date_label,
            "account":  self._account_label,
            "mynumber": self._mynumber_label,
            "other":    self._other_label,
        }

    def get_anon_label(self, original_text: str, entity_label: str) -> str:
        category = self.CATEGORY_MAP.get(entity_label, "other")
        generator = self._generators.get(category, self._other_label)
        if category not in self._cache:
            self._cache[category] = {}
            self._counters[category] = 0
        key = original_text.strip()
        if key in self._cache[category]:
            return self._cache[category][key]
        idx = self._counters[category]
        anon = generator(idx)
        self._cache[category][key] = anon
        self._counters[category] = idx + 1
        return anon

    def get_mapping_table(self) -> list[dict[str, str]]:
        rows = []
        for cat, mapping in self._cache.items():
            for original, anon in mapping.items():
                rows.append({
                    "カテゴリ": self.CATEGORY_JP.get(cat, cat),
                    "元テキスト": original,
                    "匿名ラベル": anon,
                })
        return rows


# =====================================================================
# 3. spaCy キャッシュ（現在は無効）
# =====================================================================

@st.cache_resource
def load_nlp():
    return None


# =====================================================================
# 4. 日本語フォント読み込み
# =====================================================================

@st.cache_resource
def load_font(size: int = 20):
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/OTF/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
    ]
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ja", "--format=%{file}\n"],
            capture_output=True, text=True, timeout=3
        )
        fc_fonts = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        candidates = fc_fonts + candidates
    except Exception:
        pass
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def get_sized_font(base_font, size: int):
    from PIL import ImageFont
    if hasattr(base_font, "path"):
        try:
            return ImageFont.truetype(base_font.path, size)
        except Exception:
            pass
    if hasattr(base_font, "font_variant"):
        try:
            return base_font.font_variant(size=size)
        except Exception:
            pass
    return base_font


# =====================================================================
# 5. OCR
# =====================================================================

def run_ocr(
    image: np.ndarray,
    lang: str = "jpn+eng",
    conf_threshold: float = 30.0,
) -> list[OCRBox]:
    import pytesseract
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    data = pytesseract.image_to_data(
        binary, lang=lang,
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT,
    )
    boxes: list[OCRBox] = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = float(data["conf"][i])
        if txt and conf >= conf_threshold:
            boxes.append(OCRBox(
                text=txt,
                left=data["left"][i], top=data["top"][i],
                width=data["width"][i], height=data["height"][i],
                conf=conf,
            ))
    return boxes


# =====================================================================
# 6. 行結合
# =====================================================================

def merge_boxes_into_lines(
    boxes: list[OCRBox],
    y_threshold: int = 15,
    x_gap_max: int = 80,
) -> list[tuple[str, list[OCRBox]]]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b.top + b.height // 2, b.left))
    lines: list[list[OCRBox]] = []
    current: list[OCRBox] = [sorted_boxes[0]]
    for box in sorted_boxes[1:]:
        prev = current[-1]
        cy_cur = box.top + box.height // 2
        cy_prev = prev.top + prev.height // 2
        if abs(cy_cur - cy_prev) <= y_threshold and (box.left - prev.right) <= x_gap_max:
            current.append(box)
        else:
            lines.append(current)
            current = [box]
    lines.append(current)
    return [("".join(b.text for b in lb), lb) for lb in lines]


# =====================================================================
# 7. 検出ロジック
# =====================================================================

TARGET_NER_LABELS: set[str] = {
    "Person", "PERSON",
    "ORG", "Organization",
    "LOC", "Location", "GPE", "FAC", "Facility",
}

PHONE_RE_STRICT = re.compile(
    r"(?<!\d)0\d{1,4}[-ーI\-−‐]\d{1,4}[-ーI\-−‐]\d{3,4}(?!\d)"
)
EMAIL_RE_STRICT = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)
FAX_LINE_PATTERN = re.compile(r"FAX|ＦＡＸ|Fax|fax", re.IGNORECASE)

REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("電話番号",       PHONE_RE_STRICT),
    ("郵便番号",       re.compile(r"〒?\d{3}[-ー]\d{4}")),
    ("メール",         EMAIL_RE_STRICT),
    ("マイナンバー",   re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("日付(生年月日)", re.compile(
        r"(昭和|平成|令和|S|H|R)?\s?\d{1,4}\s?[年/\-\.]\s?\d{1,2}\s?[月/\-\.]\s?\d{1,2}\s?日?"
    )),
    ("口座番号",       re.compile(r"(普通|当座)\s?\d{7,8}")),
]

FORM_LABEL_RULES: dict[str, str] = {
    "担当者名": "Person",
    "氏名":     "Person",
    "住所":     "住所",
    "所在地":   "住所",
    "電話番号": "電話番号",
    "メールアドレス": "メール",
    "団体名":   "ORG",
}

ANCHOR_STOP_WORDS = [
    "FAX", "ＦＡＸ", "Fax",
    "役職名", "団体名", "出展名",
    "担当者名", "氏名", "電話番号", "メールアドレス",
    "住所", "所在地",
]


def detect_regex(text: str) -> list[tuple[str, str, int, int]]:
    results = []
    for label, pat in REGEX_PATTERNS:
        for m in pat.finditer(text):
            results.append((m.group(), label, m.start(), m.end()))
    return results


def detect_ner(nlp, text: str, labels: set[str]) -> list[tuple[str, str, int, int]]:
    if not text.strip() or nlp is None:
        return []
    doc = nlp(text)
    return [(e.text, e.label_, e.start_char, e.end_char)
            for e in doc.ents if e.label_ in labels]


def detect_form_anchors(text: str) -> list[tuple[str, str, int, int]]:
    if not text.strip():
        return []
    txt = text.replace(" ", "").replace("\u3000", "")
    results: list[tuple[str, str, int, int]] = []
    for anchor, out_label in FORM_LABEL_RULES.items():
        pos = txt.find(anchor)
        if pos == -1:
            continue
        value_start = pos + len(anchor)
        raw_tail = txt[value_start:]
        tail = re.sub(r"^[\s:：\-ー_.・]+", "", raw_tail)
        value_start += (len(raw_tail) - len(tail))
        if not tail:
            continue
        if out_label == "電話番号":
            m = PHONE_RE_STRICT.search(tail)
            if m:
                results.append((m.group(), out_label,
                                 value_start + m.start(), value_start + m.end()))
            continue
        if out_label == "メール":
            m = EMAIL_RE_STRICT.search(tail)
            if m:
                results.append((m.group(), out_label,
                                 value_start + m.start(), value_start + m.end()))
            continue
        stop = len(tail)
        for sw in ANCHOR_STOP_WORDS:
            p = tail.find(sw)
            if p != -1 and p > 0:
                stop = min(stop, p)
        candidate = tail[:stop]
        candidate = re.sub(r"[^\u0020-\u007E\u3000-\u9FFFー]", "", candidate).strip()
        candidate = candidate.strip(":：- \t")
        if 2 <= len(candidate) <= 30:
            s = txt.find(candidate, value_start)
            if s == -1:
                s, e = value_start, value_start + len(candidate)
            else:
                e = s + len(candidate)
            results.append((candidate, out_label, s, e))
    return results


# =====================================================================
# 8. 重複除去
# =====================================================================

def _iou(a: MaskRegion, b: MaskRegion) -> float:
    ix1, iy1 = max(a.left, b.left), max(a.top, b.top)
    ix2, iy2 = min(a.right, b.right), min(a.bottom, b.bottom)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def deduplicate_regions(
    regions: list[MaskRegion],
    iou_threshold: float = 0.5,
) -> list[MaskRegion]:
    priority = {"anchor": 0, "ner": 1, "regex": 2, "manual": 3}
    sorted_regions = sorted(
        regions, key=lambda r: (priority.get(r.source, 9), r.top, r.left)
    )
    kept: list[MaskRegion] = []
    suppressed = [False] * len(sorted_regions)
    for i, r in enumerate(sorted_regions):
        if suppressed[i]:
            continue
        kept.append(r)
        for j in range(i + 1, len(sorted_regions)):
            if not suppressed[j] and _iou(r, sorted_regions[j]) >= iou_threshold:
                suppressed[j] = True
    return kept


# =====================================================================
# 9. テキスト位置 → 画像座標マッピング
# =====================================================================

def map_entity_to_boxes(
    entity_text: str, entity_label: str, anon_text: str,
    char_start: int, char_end: int,
    line_boxes: list[OCRBox], source: str = "ner",
) -> Optional[MaskRegion]:
    cum = 0
    ranges = []
    for b in line_boxes:
        s, e = cum, cum + len(b.text)
        ranges.append((s, e))
        cum = e
    matched = [line_boxes[i] for i, (s, e) in enumerate(ranges)
               if s < char_end and e > char_start]
    if not matched:
        return None
    margin = 5
    return MaskRegion(
        original_text=entity_text, label=entity_label, anon_text=anon_text,
        left=max(min(b.left for b in matched) - margin, 0),
        top=max(min(b.top for b in matched) - margin, 0),
        right=max(b.right for b in matched) + margin,
        bottom=max(b.bottom for b in matched) + margin,
        source=source,
    )


# =====================================================================
# 10. 描画：匿名ラベル
# =====================================================================

def render_anon_labels(
    image: np.ndarray,
    regions: list[MaskRegion],
    font,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (220, 40, 40),
    border_color: tuple[int, int, int] = (220, 40, 40),
) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    for r in regions:
        x1, y1, x2, y2 = r.left, r.top, r.right, r.bottom
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=border_color, width=2)
        label = r.anon_text
        region_w = max(x2 - x1 - 6, 4)
        region_h = max(y2 - y1 - 4, 4)
        best_font = font
        for sz in range(max(region_h - 2, 8), 7, -1):
            trial = get_sized_font(font, sz)
            try:
                bbox = trial.getbbox(label)
                if (bbox[2] - bbox[0]) <= region_w and (bbox[3] - bbox[1]) <= region_h:
                    best_font = trial
                    break
            except Exception:
                break
        try:
            bbox = best_font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = region_w // 2, region_h // 2
        draw.text(
            (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2),
            label, font=best_font, fill=text_color,
        )
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =====================================================================
# 11. 描画：モザイク（手動領域のみ）
# =====================================================================

def apply_mosaic(
    image: np.ndarray,
    regions: list[MaskRegion],
    mosaic_type: str = "mosaic",
    block_size: int = 15,
) -> np.ndarray:
    result = image.copy()
    for r in regions:
        if r.source != "manual":
            continue
        x1 = max(r.left, 0)
        y1 = max(r.top, 0)
        x2 = min(r.right, image.shape[1])
        y2 = min(r.bottom, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        if mosaic_type == "black":
            result[y1:y2, x1:x2] = 0
        else:
            roi = result[y1:y2, x1:x2]
            rh, rw = roi.shape[:2]
            bw, bh = max(block_size, 1), max(block_size, 1)
            small = cv2.resize(roi, (max(rw // bw, 1), max(rh // bh, 1)),
                               interpolation=cv2.INTER_LINEAR)
            result[y1:y2, x1:x2] = cv2.resize(
                small, (rw, rh), interpolation=cv2.INTER_NEAREST
            )
    return result


# =====================================================================
# 12. プレビュー画像生成（確定済み領域＋選択中マーカー<span class="cursor">█</span>

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
# 12. プレビュー画像生成（確定済み領域＋選択中マーカーを重ねる）
# =====================================================================

def draw_selection_overlay(
    image_rgb: np.ndarray,
    point1: Optional[tuple[int, int]],
    point2: Optional[tuple[int, int]],
    confirmed_regions: list[MaskRegion],
    display_w: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw

    scale  = display_w / orig_w
    disp_h = int(image_rgb.shape[0] * scale)

    # numpy でリサイズ（真っ黒防止）
    resized  = cv2.resize(image_rgb, (display_w, disp_h),
                          interpolation=cv2.INTER_LINEAR)
    pil_base = PILImage.fromarray(resized)
    overlay  = PILImage.new("RGBA", (display_w, disp_h), (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    # 確定済み領域（赤い半透明）
    for r in confirmed_regions:
        rx1 = int(r.left   * scale)
        ry1 = int(r.top    * scale)
        rx2 = int(r.right  * scale)
        ry2 = int(r.bottom * scale)
        draw.rectangle([rx1, ry1, rx2, ry2],
                       fill=(255, 0, 0, 90), outline=(255, 0, 0, 220), width=2)

    # 1点目（緑の十字）
    if point1 is not None:
        px  = int(point1[0] * scale)
        py  = int(point1[1] * scale)
        arm = 12
        draw.line([(px - arm, py), (px + arm, py)], fill=(0, 220, 0, 255), width=3)
        draw.line([(px, py - arm), (px, py + arm)], fill=(0, 220, 0, 255), width=3)
        draw.ellipse([(px - 6, py - 6), (px + 6, py + 6)],
                     outline=(0, 220, 0, 255), width=2)

    # 2点プレビュー矩形（青）
    if point1 is not None and point2 is not None:
        bx1 = int(min(point1[0], point2[0]) * scale)
        by1 = int(min(point1[1], point2[1]) * scale)
        bx2 = int(max(point1[0], point2[0]) * scale)
        by2 = int(max(point1[1], point2[1]) * scale)
        draw.rectangle([bx1, by1, bx2, by2],
                       fill=(0, 100, 255, 60), outline=(0, 100, 255, 220), width=2)

    composite = PILImage.alpha_composite(
        pil_base.convert("RGBA"), overlay
    ).convert("RGB")
    return np.array(composite)


# =====================================================================
# 13. メインパイプライン
# =====================================================================

def process_image(
    image: np.ndarray,
    nlp,
    anon_mgr: AnonymizationManager,
    use_ner: bool = True,
    use_regex: bool = True,
    target_labels: set[str] | None = None,
    ocr_lang: str = "jpn+eng",
    conf_threshold: float = 30.0,
    font=None,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (220, 40, 40),
) -> tuple[np.ndarray, list[MaskRegion], list[tuple[str, list[OCRBox]]]]:

    if target_labels is None:
        target_labels = TARGET_NER_LABELS

    boxes = run_ocr(image, lang=ocr_lang, conf_threshold=conf_threshold)
    if not boxes:
        return image.copy(), [], []

    lines = merge_boxes_into_lines(boxes)
    all_regions: list[MaskRegion] = []

    for line_text, line_boxes in lines:
        entities: list[tuple[str, str, int, int, str]] = []
        is_fax_line = bool(FAX_LINE_PATTERN.search(line_text))

        if not is_fax_line:
            anchor_hits = detect_form_anchors(line_text)
            entities += [(t, l, s, e, "anchor") for t, l, s, e in anchor_hits]

        if use_ner and nlp is not None:
            ner_hits = detect_ner(nlp, line_text, target_labels)
            entities += [(t, l, s, e, "ner") for t, l, s, e in ner_hits]

        if use_regex:
            regex_hits = detect_regex(line_text)
            if is_fax_line:
                regex_hits = [x for x in regex_hits if x[1] != "電話番号"]
            entities += [(t, l, s, e, "regex") for t, l, s, e in regex_hits]

        for ent_text, ent_label, cs, ce, src in entities:
            anon = anon_mgr.get_anon_label(ent_text, ent_label)
            region = map_entity_to_boxes(
                ent_text, ent_label, anon, cs, ce, line_boxes, source=src
            )
            if region:
                all_regions.append(region)

    all_regions = deduplicate_regions(all_regions)
    if font is None:
        font = load_font(20)
    result = render_anon_labels(image, all_regions, font, bg_color, text_color)
    return result, all_regions, lines


# =====================================================================
# 14. 前処理・スケール戻し
# =====================================================================

def preprocess_image(image_bgr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale != 1.0:
        h, w = image_bgr.shape[:2]
        return cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return image_bgr


def scale_regions_back(regions: list[MaskRegion], scale: float) -> list[MaskRegion]:
    if scale == 1.0:
        return regions
    return [MaskRegion(
        original_text=r.original_text, label=r.label, anon_text=r.anon_text,
        left=int(r.left / scale), top=int(r.top / scale),
        right=int(r.right / scale), bottom=int(r.bottom / scale),
        source=r.source,
    ) for r in regions]


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# =====================================================================
# 15. Streamlit メイン
# =====================================================================

def main() -> None:
    st.set_page_config(
        page_title="画像プライバシーマスキング",
        page_icon="🔒",
        layout="wide",
    )

    errors = _check_dependencies()
    if errors:
        st.error("## ⚠️ セットアップが必要です")
        for e in errors:
            st.markdown(e)
        st.stop()

    st.title("🔒 画像プライバシー匿名化ツール")
    st.caption("自動検出 ＋ 2点指定モザイク追加。PC はダブルクリック、iPhone は長押しで指定。")

    # ----------------------------------------------------------------
    # サイドバー
    # ----------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 設定")

        st.subheader("検出方法")
        use_ner   = st.checkbox("NER（固有表現認識）", value=False, disabled=True)
        use_regex = st.checkbox("正規表現ルール", value=True)
        if not use_ner and not use_regex:
            st.warning("少なくとも一方を有効にしてください。")
        st.caption("現在はNERを一時停止。項目名ベース + 正規表現で動作します。")
        selected = []

        st.subheader("OCR 設定")
        ocr_lang = st.text_input("Tesseract 言語コード", value="jpn+eng")
        conf_threshold = st.slider(
            "OCR 信頼度の閾値", min_value=0, max_value=90, value=30, step=5
        )
        scale_factor = st.select_slider(
            "画像スケール（大きいほど精度↑）",
            options=[0.5, 0.75, 1.0, 1.5, 2.0], value=1.0,
        )

        st.subheader("匿名ラベルの見た目")
        bg_hex  = st.color_picker("背景色", "#FFFFFF")
        txt_hex = st.color_picker("文字色", "#DC2828")
        bg_color   = hex_to_rgb(bg_hex)
        text_color = hex_to_rgb(txt_hex)

        st.subheader("手動モザイク設定")
        mosaic_type = st.radio(
            "モザイクの種類",
            options=["mosaic", "black"],
            format_func=lambda x: "🟫 モザイク（ぼかし）" if x == "mosaic" else "⬛ 黒塗り",
        )
        mosaic_block = st.slider(
            "モザイクの粗さ", min_value=5, max_value=40, value=15, step=5
        )

        st.divider()
        with st.expander("🔧 Tesseract パス設定（Windows）"):
            tess_path = st.text_input(
                "パス", value=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            if st.button("パスを適用"):
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = tess_path
                st.success("適用しました。")

    nlp  = load_nlp()
    font = load_font(20)

    # ----------------------------------------------------------------
    # アップロード
    # ----------------------------------------------------------------
    uploaded = st.file_uploader(
        "画像をアップロード（複数可）",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("👆 画像ファイルをアップロードしてください")
        st.markdown("""
**使い方:**
1. 画像をアップロード
2. 自動検出結果を確認・チェック調整
3. 検出漏れは画像を **2回操作** してモザイク範囲を指定
   - 💻 PC：**ダブルクリック**で1点目 → **ダブルクリック**で2点目
   - 📱 iPhone：**長押し**で1点目 → **長押し**で2点目
4. ダウンロード
""")
        st.stop()

    anon_mgr = AnonymizationManager()
    results  = []
    progress_bar = st.progress(0, text="処理中…")

    for idx, file in enumerate(uploaded):
        progress_bar.progress(
            idx / len(uploaded),
            text=f"処理中… ({idx + 1}/{len(uploaded)})",
        )
        from PIL import Image as PILImage
        pil_image = PILImage.open(file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        scaled_bgr = preprocess_image(image_bgr, scale_factor)
        _, regions_scaled, lines = process_image(
            scaled_bgr, nlp, anon_mgr,
            use_ner=use_ner, use_regex=use_regex,
            target_labels=set(selected) if selected else TARGET_NER_LABELS,
            ocr_lang=ocr_lang, conf_threshold=float(conf_threshold),
            font=font, bg_color=bg_color, text_color=text_color,
        )
        regions_orig = scale_regions_back(regions_scaled, scale_factor)
        results.append((file.name, pil_image, image_bgr, regions_orig, lines))

    progress_bar.progress(1.0, text="✅ 完了")

    # ----------------------------------------------------------------
    # 結果表示ループ
    # ----------------------------------------------------------------
    for fname, original_pil, original_bgr, regions, lines in results:
        st.markdown(f"---\n### 📄 {fname}")

        orig_h, orig_w = original_bgr.shape[:2]

        key_manual   = f"manual_{fname}"
        key_pt1      = f"pt1_{fname}"
        key_pt2      = f"pt2_{fname}"
        key_click_ct = f"click_ct_{fname}"   # クリック回数カウンタ

        for k, v in [
            (key_manual,   []),
            (key_pt1,      None),
            (key_pt2,      None),
            (key_click_ct, 0),
        ]:
            if k not in st.session_state:
                st.session_state[k] = v

        # ============================================================
        # STEP 1: 自動検出チェック
        # ============================================================
        st.markdown("#### ✅ STEP 1：自動検出の確認")
        if regions:
            with st.expander(
                f"🔍 自動検出結果（{len(regions)} 件）— チェックを外すと変換しません",
                expanded=True,
            ):
                st.caption("FAX番号・誤検出はチェックを外してください。")
                selected_regions: list[MaskRegion] = []
                for i, r in enumerate(regions):
                    method_name = {
                        "anchor": "📌 項目名ベース",
                        "ner":    "🧠 NER",
                        "regex":  "🔍 正規表現",
                    }.get(r.source, r.source)
                    keep = st.checkbox(
                        f"{method_name}｜[{r.label}]　{r.original_text}　→　{r.anon_text}",
                        value=True,
                        key=f"keep_{fname}_{i}",
                    )
                    if keep:
                        selected_regions.append(r)
        else:
            selected_regions = []
            st.warning("⚠️ 自動検出ゼロ。STEP 2 で手動追加してください。")

        # ============================================================
        # STEP 2: 2点指定でモザイク追加
        # ============================================================
        st.markdown("#### 🖊️ STEP 2：検出漏れをモザイク追加")

        # 操作ガイド（状態に応じて変化）
        pt1 = st.session_state[key_pt1]
        pt2 = st.session_state[key_pt2]

        if pt1 is None:
            st.info(
                "💻 PC：隠したい範囲の **左上をダブルクリック**\n\n"
                "📱 iPhone：隠したい範囲の **左上を長押し**"
            )
        elif pt2 is None:
            st.info(
                f"✅ 1点目 ({pt1[0]}, {pt1[1]}) を記録しました\n\n"
                "💻 PC：範囲の **右下をダブルクリック**\n\n"
                "📱 iPhone：範囲の **右下を長押し**"
            )
        else:
            st.success(
                f"✅ 範囲プレビュー: "
                f"({min(pt1[0],pt2[0])}, {min(pt1[1],pt2[1])}) 〜 "
                f"({max(pt1[0],pt2[0])}, {max(pt1[1],pt2[1])})\n\n"
                "↓「この範囲をモザイクに追加」ボタンを押してください"
            )

        # プレビュー画像生成
        auto_masked_bgr = render_anon_labels(
            original_bgr, selected_regions, font, bg_color, text_color
        )
        auto_masked_rgb = cv2.cvtColor(auto_masked_bgr, cv2.COLOR_BGR2RGB)

        DISPLAY_W = 680

        preview_img = draw_selection_overlay(
            auto_masked_rgb,
            st.session_state[key_pt1],
            st.session_state[key_pt2],
            st.session_state[key_manual],
            display_w=DISPLAY_W,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        # ---- streamlit-image-coordinates で座標取得 ----
        # use_column_width=True にして表示幅を DISPLAY_W に揃える
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            from PIL import Image as PILImage


            coords = streamlit_image_coordinates(
                PILImg.fromarray(preview_img),
                key=f"coords_{fname}",
            )

            if coords is not None:
                # 表示座標 → 元画像座標へ変換
                scale_back = orig_w / DISPLAY_W
                real_x = int(coords["x"] * scale_back)
                real_y = int(coords["y"] * scale_back)

                # クリック回数で1点目・2点目を切り替え
                ct = st.session_state[key_click_ct]

                # 同じ座標の連続反応を無視（誤検知防止）
                prev_pt = st.session_state[key_pt1] if ct % 2 == 1 else None
                is_same = (prev_pt is not None and
                           abs(prev_pt[0] - real_x) < 5 and
                           abs(prev_pt[1] - real_y) < 5)

                if not is_same:
                    if st.session_state[key_pt1] is None:
                        st.session_state[key_pt1] = (real_x, real_y)
                        st.session_state[key_click_ct] = ct + 1
                        st.rerun()
                    elif st.session_state[key_pt2] is None:
                        st.session_state[key_pt2] = (real_x, real_y)
                        st.session_state[key_click_ct] = ct + 1
                        st.rerun()

        except ImportError:
            st.warning(
                "⚠️ `streamlit-image-coordinates` が未インストールです。\n"
                "`requirements.txt` に `streamlit-image-coordinates` を追加して"
                "再デプロイしてください。"
            )

        # ---- ボタン3つ ----
        col_add, col_reset1, col_reset_all = st.columns(3)

        with col_add:
            if pt1 is not None and pt2 is not None:
                if st.button(
                    "✅ この範囲をモザイクに追加",
                    key=f"confirm_{fname}",
                    use_container_width=True,
                ):
                    new_r = MaskRegion(
                        original_text=f"manual_{len(st.session_state[key_manual])}",
                        label="手動",
                        anon_text="",
                        left=min(pt1[0], pt2[0]),
                        top=min(pt1[1], pt2[1]),
                        right=max(pt1[0], pt2[0]),
                        bottom=max(pt1[1], pt2[1]),
                        source="manual",
                    )
                    st.session_state[key_manual].append(new_r)
                    st.session_state[key_pt1]      = None
                    st.session_state[key_pt2]      = None
                    st.session_state[key_click_ct] = 0
                    st.rerun()

        with col_reset1:
            if st.button(
                "↩️ 選択をやり直す",
                key=f"reset1_{fname}",
                use_container_width=True,
            ):
                st.session_state[key_pt1]      = None
                st.session_state[key_pt2]      = None
                st.session_state[key_click_ct] = 0
                st.rerun()

        with col_reset_all:
            if st.button(
                "🗑️ 手動追加を全削除",
                key=f"reset_all_{fname}",
                use_container_width=True,
            ):
                st.session_state[key_manual]   = []
                st.session_state[key_pt1]      = None
                st.session_state[key_pt2]      = None
                st.session_state[key_click_ct] = 0
                st.rerun()

        # ---- 追加済みリスト ----
        manual_regions: list[MaskRegion] = st.session_state[key_manual]
        if manual_regions:
            with st.expander(
                f"📋 追加済み手動モザイク（{len(manual_regions)} 件）",
                expanded=False,
            ):
                for mi, mr in enumerate(manual_regions):
                    col_i, col_d = st.columns([5, 1])
                    with col_i:
                        st.caption(
                            f"#{mi + 1}　"
                            f"左:{mr.left} 上:{mr.top} 右:{mr.right} 下:{mr.bottom}"
                        )
                    with col_d:
                        if st.button("🗑️", key=f"del_{fname}_{mi}"):
                            st.session_state[key_manual].pop(mi)
                            st.rerun()

        # ============================================================
        # STEP 3: 最終画像生成・表示・ダウンロード
        # ============================================================
        st.markdown("#### 🖼️ STEP 3：最終プレビュー＆ダウンロード")

        final_bgr = render_anon_labels(
            original_bgr, selected_regions, font, bg_color, text_color
        )
        final_bgr = apply_mosaic(
            final_bgr, manual_regions,
            mosaic_type=mosaic_type, block_size=mosaic_block,
        )
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**元画像**")
            st.image(original_pil, use_container_width=True)
        with col2:
            st.markdown("**匿名化後（自動＋手動モザイク）**")
            st.image(final_rgb, use_container_width=True)

        with st.expander("📝 OCR テキスト（デバッグ用）"):
            for line_text, _ in lines:
                if line_text.strip():
                    st.text(line_text)

        from PIL import Image as PILImage
        buf = BytesIO()
        PILImage.fromarray(final_rgb).save(buf, format="PNG")
        st.download_button(
            label=f"⬇️ {fname} の匿名化画像をダウンロード",
            data=buf.getvalue(),
            file_name=f"anon_{fname.rsplit('.', 1)[0]}.png",
            mime="image/png",
            key=f"dl_{fname}",
        )

    # ----------------------------------------------------------------
    # マッピングテーブル
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 匿名化マッピング一覧")
    st.caption("⚠️ この対応表は自分専用で保管し、第三者には絶対に共有しないでください。")
    mapping = anon_mgr.get_mapping_table()
    if mapping:
        st.table(mapping)
        text_buf = io.StringIO()
        writer = csv.DictWriter(
            text_buf, fieldnames=["カテゴリ", "元テキスト", "匿名ラベル"]
        )
        writer.writeheader()
        writer.writerows(mapping)
        st.download_button(
            label="⬇️ マッピング表を CSV でダウンロード",
            data=text_buf.getvalue().encode("utf-8-sig"),
            file_name="anonymization_mapping.csv",
            mime="text/csv",
        )
    else:
        st.info("検出された固有表現がないため、マッピングはありません。")


if __name__ == "__main__":
    main()

"""
app.py – 画像内の個人情報を匿名化ラベルで置換する Streamlit アプリ
====================================================================
・人名      → A, B, C, ... Z, AA, AB, ...
・地名      → 地点あ, 地点い, 地点う, ...
・電話番号  → 000-0000-0001, 000-0000-0002, ...
・郵便番号  → 〒000-0001, 〒000-0002, ...
・メール    → anonymous001@example.com, ...
・組織      → 組織ア, 組織イ, 組織ウ, ...
・その他    → XXX-001, XXX-002, ...

同一の元テキストには必ず同じ仮名ラベルが割り当てられます。

使い方: streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# =====================================================================
# 0. 起動時チェック（Tesseract / spaCy / GiNZA）
# =====================================================================

def _check_dependencies() -> list[str]:
    """不足している依存関係のリストを返す。空なら全OK。"""
    errors: list[str] = []

    # Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        errors.append(
            "❌ **Tesseract が見つかりません。**\n"
            "- Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-jpn`\n"
            "- macOS: `brew install tesseract tesseract-lang`\n"
            "- Windows: [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki) からインストール後、"
            "`pytesseract.pytesseract.tesseract_cmd` にパスを設定してください。"
        )

    # spaCy + GiNZA
    try:
        import spacy
        spacy.load("ja_ginza")
    except OSError:
        errors.append(
            "❌ **GiNZA モデルが見つかりません。**\n"
            "`pip install ja-ginza` を実行してください。\n"
            "（重いモデルが必要な場合: `pip install ja-ginza-electra`）"
        )
    except ImportError:
        errors.append(
            "❌ **spaCy がインストールされていません。**\n"
            "`pip install spacy ginza ja-ginza` を実行してください。"
        )

    return errors


# =====================================================================
# 1. データクラス
# =====================================================================

@dataclass
class OCRBox:
    """Tesseract が返す 1 単語ぶんのバウンディングボックス."""
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
    """匿名化対象領域."""
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
    """
    同一エンティティに同一の仮名ラベルを割り当てる。
    カテゴリごとに独立したカウンタを持つ。
    """

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
        self._generators: dict[str, callable] = {
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
# 3. spaCy (GiNZA) キャッシュ読み込み
# =====================================================================

@st.cache_resource(show_spinner="spaCy (GiNZA) モデルを読み込み中…")
def load_nlp():
    import spacy
    return spacy.load("ja_ginza")


# =====================================================================
# 4. 日本語フォント読み込み（PIL 描画用）
# =====================================================================

@st.cache_resource
def load_font(size: int = 20):
    from PIL import ImageFont

    # 静的候補
    candidates = [
        # Linux (Ubuntu / Debian)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/OTF/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Windows
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothR.ttc",
    ]

    # fc-list による動的探索（Linux / macOS）
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

    # フォールバック（サイズ無効だが動く）
    return ImageFont.load_default()


def get_sized_font(base_font, size: int):
    """フォントを指定サイズで再取得する。失敗時は base_font を返す。"""
    from PIL import ImageFont
    if hasattr(base_font, "path"):
        try:
            return ImageFont.truetype(base_font.path, size)
        except Exception:
            pass
    # font_variant が使えるか試す
    if hasattr(base_font, "font_variant"):
        try:
            return base_font.font_variant(size=size)
        except Exception:
            pass
    return base_font


# =====================================================================
# 5. OCR
# =====================================================================

def run_ocr(image: np.ndarray, lang: str = "jpn+eng", conf_threshold: float = 30.0) -> list[OCRBox]:
    import pytesseract
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 適応的二値化でコントラストを高める
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
# 7. NER + 正規表現検出
# =====================================================================

TARGET_NER_LABELS: set[str] = {
    "Person", "PERSON",
    "ORG", "Organization",
    "LOC", "Location", "GPE", "FAC", "Facility",
}

}

REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("電話番号",       re.compile(r"0\d{1,4}[-ー]?\d{1,4}[-ー]?\d{3,4}")),
    ("郵便番号",       re.compile(r"〒?\d{3}[-ー]\d{4}")),
    ("メール",         re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("マイナンバー",   re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("日付(生年月日)", re.compile(
        r"(昭和|平成|令和|S|H|R)?\s?\d{1,4}\s?[年/\-\.]\s?\d{1,2}\s?[月/\-\.]\s?\d{1,2}\s?日?"
    )),
    ("口座番号",       re.compile(r"(普通|当座)\s?\d{7,8}")),
]


def detect_ner(nlp, text: str, labels: set[str]) -> list[tuple[str, str, int, int]]:
    if not text.strip():
        return []
    doc = nlp(text)
    return [(e.text, e.label_, e.start_char, e.end_char)
            for e in doc.ents if e.label_ in labels]


def detect_regex(text: str) -> list[tuple[str, str, int, int]]:
    results = []
    for label, pat in REGEX_PATTERNS:
        for m in pat.finditer(text):
            results.append((m.group(), label, m.start(), m.end()))
    return results


PHONE_RE_STRICT = re.compile(r"(?<!\d)0\d{1,4}[-ー−‐]?\d{1,4}[-ー−‐]?\d{3,4}(?!\d)")
EMAIL_RE_STRICT = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

FORM_LABEL_RULES: dict[str, str] = {
    "担当者名": "Person",
    "氏名": "Person",
    "電話番号": "電話番号",
    "メールアドレス": "メール",
}

ANCHOR_STOP_WORDS = [
    "FAX", "ＦＡＸ",
    "役職名", "団体名", "出展名",
    "担当者名", "氏名", "電話番号", "メールアドレス",
]

def detect_form_anchors(text: str) -> list[tuple[str, str, int, int]]:
    """
    定型帳票向け:
    「担当者名」「氏名」「電話番号」「メールアドレス」の
    ラベル直後の値を優先的に拾う。
    """
    if not text.strip():
        return []

    txt = text.replace(" ", "").replace("　", "")
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
                s = value_start + m.start()
                e = value_start + m.end()
                results.append((m.group(), out_label, s, e))
            continue

        if out_label == "メール":
            m = EMAIL_RE_STRICT.search(tail)
            if m:
                s = value_start + m.start()
                e = value_start + m.end()
                results.append((m.group(), out_label, s, e))
            continue

        stop = len(tail)
        for sw in ANCHOR_STOP_WORDS:
            p = tail.find(sw)
            if p != -1:
                stop = min(stop, p)

        candidate = tail[:stop]
        candidate = re.sub(r"[^0-9A-Za-zぁ-んァ-ヶ一-龥々ー・]", "", candidate).strip()

        if 2 <= len(candidate) <= 20:
            s = txt.find(candidate, value_start)
            if s != -1:
                e = s + len(candidate)
                results.append((candidate, out_label, s, e))

    return results




# =====================================================================
# 8. 重複リージョン除去（IoU ベース）
# =====================================================================

def _iou(a: MaskRegion, b: MaskRegion) -> float:
    ix1 = max(a.left, b.left)
    iy1 = max(a.top, b.top)
    ix2 = min(a.right, b.right)
    iy2 = min(a.bottom, b.bottom)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def deduplicate_regions(regions: list[MaskRegion], iou_threshold: float = 0.5) -> list[MaskRegion]:
    """IoU が閾値以上のリージョンを NER 優先でまとめる（NMS 的処理）。"""
    # NER を優先（NER を先に並べる）
    priority = {"anchor": 0, "ner": 1, "regex": 2}
    sorted_regions = sorted(regions, key=lambda r: (priority.get(r.source, 9), r.top, r.left))

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
        s = cum
        e = cum + len(b.text)
        ranges.append((s, e))
        cum = e

    matched = [line_boxes[i] for i, (s, e) in enumerate(ranges)
               if s < char_end and e > char_start]
    if not matched:
        return None

    margin = 5
    return MaskRegion(
        original_text=entity_text,
        label=entity_label,
        anon_text=anon_text,
        left=max(min(b.left for b in matched) - margin, 0),
        top=max(min(b.top for b in matched) - margin, 0),
        right=max(b.right for b in matched) + margin,
        bottom=max(b.bottom for b in matched) + margin,
        source=source,
    )


# =====================================================================
# 10. 匿名ラベル描画（PIL で日本語テキスト描画）
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
        # 領域が有効か確認
        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle([x1, y1, x2, y2], fill=bg_color, outline=border_color, width=2)

        label = r.anon_text
        region_w = max(x2 - x1 - 6, 4)
        region_h = max(y2 - y1 - 4, 4)

        # 最適フォントサイズを探索
        best_font = font
        base_size = max(region_h - 2, 8)
        for sz in range(base_size, 7, -1):
            trial = get_sized_font(font, sz)
            try:
                bbox = trial.getbbox(label)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                break
            if tw <= region_w and th <= region_h:
                best_font = trial
                break

        try:
            bbox = best_font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = region_w // 2, region_h // 2

        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 - th) // 2
        draw.text((tx, ty), label, font=best_font, fill=text_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =====================================================================
# 11. メインパイプライン
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

    from PIL import ImageFont as PILImageFont
    if target_labels is None:
        target_labels = TARGET_NER_LABELS

    boxes = run_ocr(image, lang=ocr_lang, conf_threshold=conf_threshold)
    if not boxes:
        return image.copy(), [], []

    lines = merge_boxes_into_lines(boxes)
    all_regions: list[MaskRegion] = []

    for line_text, line_boxes in lines:
                entities: list[tuple[str, str, int, int, str]] = []

        # 1) 定型帳票の項目名ベース検出（最優先）
        anchor_hits = detect_form_anchors(line_text)
        entities += [(t, l, s, e, "anchor") for t, l, s, e in anchor_hits]

        # 2) NER
        if use_ner:
            entities += [(t, l, s, e, "ner")
                         for t, l, s, e in detect_ner(nlp, line_text, target_labels)]

        # 3) 正規表現
        if use_regex:
            regex_hits = detect_regex(line_text)

            # FAX行にある電話番号は変換しない
            if "FAX" in line_text.upper() or "ＦＡＸ" in line_text:
                regex_hits = [x for x in regex_hits if x[1] != "電話番号"]

            entities += [(t, l, s, e, "regex") for t, l, s, e in regex_hits]

        for ent_text, ent_label, cs, ce, src in entities:
            anon = anon_mgr.get_anon_label(ent_text, ent_label)
            region = map_entity_to_boxes(ent_text, ent_label, anon, cs, ce, line_boxes, source=src)
            if region:
                all_regions.append(region)

    # 重複排除
    all_regions = deduplicate_regions(all_regions)

    if font is None:
        font = load_font(20)
    result = render_anon_labels(image, all_regions, font, bg_color, text_color)

    return result, all_regions, lines


# =====================================================================
# 12. 画像前処理（解像度スケール）
# =====================================================================

def preprocess_image(image_bgr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale != 1.0:
        h, w = image_bgr.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image_bgr


def scale_regions_back(regions: list[MaskRegion], scale: float) -> list[MaskRegion]:
    """スケール後の座標を元の画像サイズに戻す。"""
    if scale == 1.0:
        return regions
    scaled = []
    for r in regions:
        scaled.append(MaskRegion(
            original_text=r.original_text,
            label=r.label,
            anon_text=r.anon_text,
            left=int(r.left / scale),
            top=int(r.top / scale),
            right=int(r.right / scale),
            bottom=int(r.bottom / scale),
            source=r.source,
        ))
    return scaled


# =====================================================================
# 13. Streamlit UI
# =====================================================================

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    st.set_page_config(
        page_title="画像プライバシーマスキング",
        page_icon="🔒",
        layout="wide",
    )

    # ---- 依存チェック ----
    errors = _check_dependencies()
    if errors:
        st.error("## ⚠️ セットアップが必要です")
        for e in errors:
            st.markdown(e)
        st.stop()

    st.title("🔒 画像プライバシー匿名化ツール")
    st.caption(
        "画像内の人名・地名・電話番号等を検出し、一貫した匿名ラベルで置換します。\n"
        "同一の名前は常に同じラベルに変換されます。外部 API 不使用・完全ローカル動作。"
    )

    # ---- サイドバー ----
    with st.sidebar:
        st.header("⚙️ 設定")

        st.subheader("検出方法")
        use_ner = st.checkbox("NER（固有表現認識）", value=True)
        use_regex = st.checkbox("正規表現ルール", value=True)

        if not use_ner and not use_regex:
            st.warning("少なくとも一方を有効にしてください。")

        st.subheader("NER ラベル選択")
        all_labels = [
    "Person", "PERSON",
    "ORG", "Organization",
    "LOC", "Location", "GPE", "FAC", "Facility",
]

        ]
        selected = st.multiselect("対象ラベル", all_labels, default=all_labels)

        st.subheader("OCR 設定")
        ocr_lang = st.text_input("Tesseract 言語コード", value="jpn+eng")
        conf_threshold = st.slider(
            "OCR 信頼度の閾値（低いほど多くの文字を検出）",
            min_value=0, max_value=90, value=30, step=5,
        )
        scale_factor = st.select_slider(
            "画像スケール（大きいほど精度↑・速度↓）",
            options=[0.5, 0.75, 1.0, 1.5, 2.0],
            value=1.0,
        )

        st.subheader("匿名ラベルの見た目")
        bg_hex = st.color_picker("背景色", "#FFFFFF")
        txt_hex = st.color_picker("文字色", "#DC2828")
        bg_color = hex_to_rgb(bg_hex)
        text_color = hex_to_rgb(txt_hex)

        st.divider()
        st.markdown(
            "**匿名ラベル凡例**\n\n"
            "| カテゴリ | 形式 |\n"
            "|---|---|\n"
            "| 人名 | A, B, C … |\n"
            "| 地名 | 地点あ, 地点い … |\n"
            "| 組織 | 組織ア, 組織イ … |\n"
            "| 電話番号 | 000-0000-0001 … |\n"
            "| 郵便番号 | 〒000-0001 … |\n"
            "| メール | anonymous001@… |\n"
            "| 日付 | 0000年00月01日 … |\n"
            "| マイナンバー | 0000-0000-0001 … |\n"
        )

        st.divider()
        with st.expander("🔧 Tesseract パス設定（Windows）"):
            tess_path = st.text_input(
                "Tesseract 実行ファイルのパス",
                value=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            )
            if st.button("パスを適用"):
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = tess_path
                st.success("パスを適用しました。")

    # ---- モデル・フォント読み込み ----
    nlp = load_nlp()
    font = load_font(20)

    # ---- アップロード ----
    uploaded = st.file_uploader(
        "画像をアップロード（複数可）",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("👆 画像ファイルをアップロードしてください（PNG / JPG / TIFF 等）")
        st.markdown(
            """
            **対応している個人情報の種類:**
            - 👤 人名・氏名（GiNZA NER）
            - 🏢 組織名・会社名
            - 📍 地名・住所
            - 📞 電話番号（正規表現）
            - 📮 郵便番号
            - 📧 メールアドレス
            - 📅 生年月日・日付
            - 🔢 マイナンバー・口座番号
            """
        )
        st.stop()

    # 全画像で共有する匿名化マネージャー
    anon_mgr = AnonymizationManager()

    results = []
    progress_bar = st.progress(0, text="処理中…")

    for idx, file in enumerate(uploaded):
        progress_bar.progress((idx) / len(uploaded), text=f"「{file.name}」を処理中… ({idx + 1}/{len(uploaded)})")

        from PIL import Image as PILImage
        pil_image = PILImage.open(file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # スケール処理（OCR用）
        scaled_bgr = preprocess_image(image_bgr, scale_factor)

        masked_scaled, regions_scaled, lines = process_image(
            scaled_bgr, nlp, anon_mgr,
            use_ner=use_ner,
            use_regex=use_regex,
            target_labels=set(selected) if selected else TARGET_NER_LABELS,
            ocr_lang=ocr_lang,
            conf_threshold=float(conf_threshold),
            font=font,
            bg_color=bg_color,
            text_color=text_color,
        )

        # 元スケールに変換して描画し直す
        regions_orig = scale_regions_back(regions_scaled, scale_factor)
        masked = render_anon_labels(image_bgr, regions_orig, font, bg_color, text_color)

        results.append((file.name, pil_image, masked, regions_orig, lines))

    progress_bar.progress(1.0, text="✅ 全ファイル処理完了")

    # ---- 結果表示 ----
    for fname, original, masked, regions, lines in results:
        st.markdown(f"---\n### 📄 {fname}")

        # -----------------------------
        # 確認UI: チェックを外すと変換しない
        # -----------------------------
        if regions:
            with st.expander(f"🔍 検出・置換結果（{len(regions)} 件）", expanded=True):
                st.caption("チェックを外すと、その項目は変換しません（FAXや誤検出の除外用）。")

                selected_regions = []
                for i, r in enumerate(regions):
                    method_name = {
                        "anchor": "項目名ベース",
                        "ner": "NER",
                        "regex": "正規表現",
                    }.get(r.source, r.source)

                    keep = st.checkbox(
                        f"[{r.label}] {r.original_text} → {r.anon_text} / {method_name}",
                        value=True,
                        key=f"keep_{fname}_{i}",
                    )
                    if keep:
                        selected_regions.append(r)
        else:
            selected_regions = []
            st.warning("⚠️ 固有表現は検出されませんでした。OCR の信頼度閾値を下げてみてください。")

        # チェック後のリージョンで再描画
        original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        masked_checked = render_anon_labels(original_bgr, selected_regions, font, bg_color, text_color)
        masked_rgb = cv2.cvtColor(masked_checked, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**元画像**")
            st.image(original, use_container_width=True)
        with col2:
            st.markdown("**匿名化後（確認反映済み）**")
            st.image(masked_rgb, use_container_width=True)

        with st.expander("📝 OCR テキスト（デバッグ用）"):
            for line_text, _ in lines:
                if line_text.strip():
                    st.text(line_text)

        # ダウンロード
        buf = BytesIO()
        from PIL import Image as PILImage
        PILImage.fromarray(masked_rgb).save(buf, format="PNG")

        st.download_button(
            label=f"⬇️ {fname} の匿名化画像をダウンロード",
            data=buf.getvalue(),
            file_name=f"anon_{fname.rsplit('.', 1)[0]}.png",
            mime="image/png",
        )

    # ---- 全体マッピングテーブル ----
    st.markdown("---")
    st.subheader("📋 匿名化マッピング一覧（全画像共通）")
    st.caption("⚠️ この対応表は自分専用で保管し、第三者には絶対に共有しないでください。")

    mapping = anon_mgr.get_mapping_table()
    if mapping:
        st.table(mapping)

        text_buf = io.StringIO()
        writer = csv.DictWriter(text_buf, fieldnames=["カテゴリ", "元テキスト", "匿名ラベル"])
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

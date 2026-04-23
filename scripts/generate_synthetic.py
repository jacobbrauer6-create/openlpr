"""
scripts/generate_synthetic.py
------------------------------
Generates synthetic licence plate images with full annotations:
  - Plate text (known ground truth)
  - Bounding box in YOLO format
  - Country / region metadata

Supports: US (all 50 states), UK, Germany, France, EU generic,
          Australia, Canada, Japan, Brazil, India

Usage:
    python scripts/generate_synthetic.py --count 5000 --regions us eu
    python scripts/generate_synthetic.py --count 500 --regions us --state CA
    python scripts/generate_synthetic.py --count 10000 --regions all --augment

Output structure (YOLO-compatible, matches prepare_dataset.py expectations):
    data/raw/synthetic/
        images/  *.jpg
        labels/  *.txt   (YOLO format: 0 cx cy w h)
        manifest.json    (full annotation metadata)
"""

import argparse
import json
import math
import os
import random
import string
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Font loading — works on Windows, macOS, Linux
# ---------------------------------------------------------------------------

def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Try to load a monospace/sans font from common system locations.
    Falls back to Pillow's built-in bitmap font if nothing is found.
    Windows candidates cover most machines; Linux/macOS lists for CI.
    """
    candidates_windows = [
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
    ]
    candidates_linux = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    ]
    candidates_mac = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Courier New.ttf",
    ]

    for path in candidates_windows + candidates_linux + candidates_mac:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Last resort: Pillow default bitmap font (no size control)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Plate format definitions
# ---------------------------------------------------------------------------

@dataclass
class PlateFormat:
    country: str
    width: int
    height: int
    bg_color: tuple        # RGB plate background
    text_color: tuple      # RGB text
    border_color: tuple
    border_width: int
    font_size: int
    text_pattern: str      # used to generate plate text
    has_country_badge: bool = False
    badge_text: str = ""
    badge_color: tuple = (0, 0, 120)
    badge_text_color: tuple = (255, 255, 255)
    state_text: str = ""   # shown above/below main text on some plates
    two_line: bool = False  # some plates split into 2 lines


PLATE_FORMATS = {
    # -----------------------------------------------------------------------
    # United States — white plate, black text, varies by state
    # -----------------------------------------------------------------------
    "US_standard": PlateFormat(
        country="US", width=440, height=220,
        bg_color=(240, 240, 240), text_color=(20, 20, 20),
        border_color=(150, 150, 150), border_width=6,
        font_size=90, text_pattern="AAA####",   # e.g. ABC1234 — overridden per state
    ),
    # -----------------------------------------------------------------------
    # United Kingdom — white front / yellow rear, black text
    # -----------------------------------------------------------------------
    "UK_front": PlateFormat(
        country="UK", width=520, height=180,
        bg_color=(245, 245, 245), text_color=(10, 10, 10),
        border_color=(80, 80, 80), border_width=5,
        font_size=100, text_pattern="AA##AAA",
    ),
    "UK_rear": PlateFormat(
        country="UK", width=520, height=180,
        bg_color=(255, 213, 0), text_color=(10, 10, 10),
        border_color=(80, 80, 80), border_width=5,
        font_size=100, text_pattern="AA##AAA",
    ),
    # -----------------------------------------------------------------------
    # Germany — white, black text, blue EU badge on left
    # -----------------------------------------------------------------------
    "DE": PlateFormat(
        country="DE", width=520, height=180,
        bg_color=(248, 248, 248), text_color=(15, 15, 15),
        border_color=(60, 60, 60), border_width=5,
        font_size=88, text_pattern="AA#AA####",
        has_country_badge=True, badge_text="D",
        badge_color=(0, 51, 153), badge_text_color=(255, 204, 0),
    ),
    # -----------------------------------------------------------------------
    # France — white, blue badge left, black text
    # -----------------------------------------------------------------------
    "FR": PlateFormat(
        country="FR", width=520, height=180,
        bg_color=(255, 255, 255), text_color=(10, 10, 10),
        border_color=(70, 70, 70), border_width=5,
        font_size=90, text_pattern="AA###AA",
        has_country_badge=True, badge_text="F",
        badge_color=(0, 51, 153), badge_text_color=(255, 255, 255),
    ),
    # -----------------------------------------------------------------------
    # EU generic (Spain, Italy, Netherlands, Poland, etc.)
    # -----------------------------------------------------------------------
    "EU_generic": PlateFormat(
        country="EU", width=520, height=180,
        bg_color=(255, 255, 255), text_color=(10, 10, 10),
        border_color=(70, 70, 70), border_width=4,
        font_size=88, text_pattern="AAA####",
        has_country_badge=True, badge_text="EU",
        badge_color=(0, 51, 153), badge_text_color=(255, 255, 255),
    ),
    # -----------------------------------------------------------------------
    # Australia — white, black text, state shown above
    # -----------------------------------------------------------------------
    "AU": PlateFormat(
        country="AU", width=440, height=200,
        bg_color=(248, 248, 248), text_color=(20, 20, 20),
        border_color=(100, 100, 100), border_width=5,
        font_size=82, text_pattern="AAA###",
        two_line=True, state_text="NSW",
    ),
    # -----------------------------------------------------------------------
    # Canada — white, black text, province shown above
    # -----------------------------------------------------------------------
    "CA": PlateFormat(
        country="CA", width=440, height=210,
        bg_color=(245, 245, 245), text_color=(20, 20, 20),
        border_color=(120, 120, 120), border_width=5,
        font_size=86, text_pattern="AAA####",
        two_line=True, state_text="ONTARIO",
    ),
    # -----------------------------------------------------------------------
    # Brazil — white/grey, Mercosul format
    # -----------------------------------------------------------------------
    "BR": PlateFormat(
        country="BR", width=480, height=200,
        bg_color=(240, 240, 240), text_color=(15, 15, 15),
        border_color=(80, 80, 80), border_width=5,
        font_size=92, text_pattern="AAA#A##",
    ),
    # -----------------------------------------------------------------------
    # India — white front / yellow rear, black text
    # -----------------------------------------------------------------------
    "IN_front": PlateFormat(
        country="IN", width=500, height=180,
        bg_color=(255, 255, 255), text_color=(10, 10, 10),
        border_color=(60, 60, 60), border_width=4,
        font_size=78, text_pattern="AA##AA####",
        two_line=True, state_text="MH 12",
    ),
    "IN_rear": PlateFormat(
        country="IN", width=500, height=180,
        bg_color=(255, 220, 0), text_color=(10, 10, 10),
        border_color=(60, 60, 60), border_width=4,
        font_size=78, text_pattern="AA##AA####",
        two_line=True, state_text="MH 12",
    ),
    # -----------------------------------------------------------------------
    # Japan — white, green border for standard, black text
    # -----------------------------------------------------------------------
    "JP": PlateFormat(
        country="JP", width=440, height=220,
        bg_color=(255, 255, 255), text_color=(10, 10, 10),
        border_color=(0, 120, 0), border_width=8,
        font_size=80, text_pattern="##-##",
        two_line=True, state_text="品川",
    ),
}

# ---------------------------------------------------------------------------
# US state definitions
# ---------------------------------------------------------------------------

US_STATES = {
    "AL": {"name": "Alabama",       "pattern": "AA#####",  "color": (200, 220, 240)},
    "AK": {"name": "Alaska",        "pattern": "AAA###",   "color": (240, 240, 240)},
    "AZ": {"name": "Arizona",       "pattern": "AAA####",  "color": (210, 230, 210)},
    "AR": {"name": "Arkansas",      "pattern": "###AAA",   "color": (240, 240, 240)},
    "CA": {"name": "California",    "pattern": "#AAA###",  "color": (240, 240, 240)},
    "CO": {"name": "Colorado",      "pattern": "AAA-###",  "color": (220, 235, 255)},
    "CT": {"name": "Connecticut",   "pattern": "AA#####",  "color": (240, 240, 240)},
    "DE": {"name": "Delaware",      "pattern": "######",   "color": (240, 240, 240)},
    "FL": {"name": "Florida",       "pattern": "AAA-A##",  "color": (240, 250, 240)},
    "GA": {"name": "Georgia",       "pattern": "AAA####",  "color": (240, 240, 240)},
    "HI": {"name": "Hawaii",        "pattern": "AAA###",   "color": (255, 240, 220)},
    "ID": {"name": "Idaho",         "pattern": "AA####A",  "color": (240, 240, 240)},
    "IL": {"name": "Illinois",      "pattern": "AA#####",  "color": (240, 240, 240)},
    "IN": {"name": "Indiana",       "pattern": "###AAA",   "color": (240, 240, 240)},
    "IA": {"name": "Iowa",          "pattern": "AAA####",  "color": (240, 240, 240)},
    "KS": {"name": "Kansas",        "pattern": "###AAA",   "color": (240, 240, 240)},
    "KY": {"name": "Kentucky",      "pattern": "###AAA",   "color": (240, 240, 240)},
    "LA": {"name": "Louisiana",     "pattern": "AAA###",   "color": (240, 240, 240)},
    "ME": {"name": "Maine",         "pattern": "####AA",   "color": (240, 240, 240)},
    "MD": {"name": "Maryland",      "pattern": "AA#####",  "color": (240, 240, 240)},
    "MA": {"name": "Massachusetts", "pattern": "####AA",   "color": (240, 240, 240)},
    "MI": {"name": "Michigan",      "pattern": "AAA####",  "color": (240, 240, 240)},
    "MN": {"name": "Minnesota",     "pattern": "AAA###",   "color": (240, 240, 240)},
    "MS": {"name": "Mississippi",   "pattern": "AAA###",   "color": (240, 240, 240)},
    "MO": {"name": "Missouri",      "pattern": "AA#-A##",  "color": (240, 240, 240)},
    "MT": {"name": "Montana",       "pattern": "######",   "color": (240, 240, 240)},
    "NE": {"name": "Nebraska",      "pattern": "AA#####",  "color": (240, 240, 240)},
    "NV": {"name": "Nevada",        "pattern": "###AAA",   "color": (240, 240, 240)},
    "NH": {"name": "New Hampshire", "pattern": "#######",  "color": (240, 240, 240)},
    "NJ": {"name": "New Jersey",    "pattern": "AA##A",    "color": (240, 240, 240)},
    "NM": {"name": "New Mexico",    "pattern": "AAA###",   "color": (240, 250, 230)},
    "NY": {"name": "New York",      "pattern": "AAA####",  "color": (240, 240, 240)},
    "NC": {"name": "North Carolina","pattern": "AAA####",  "color": (240, 240, 240)},
    "ND": {"name": "North Dakota",  "pattern": "AAA###",   "color": (240, 240, 240)},
    "OH": {"name": "Ohio",          "pattern": "AAA####",  "color": (240, 240, 240)},
    "OK": {"name": "Oklahoma",      "pattern": "AAA###",   "color": (240, 240, 240)},
    "OR": {"name": "Oregon",        "pattern": "AAA###",   "color": (240, 240, 240)},
    "PA": {"name": "Pennsylvania",  "pattern": "AAA####",  "color": (240, 240, 240)},
    "RI": {"name": "Rhode Island",  "pattern": "AA####",   "color": (240, 240, 240)},
    "SC": {"name": "South Carolina","pattern": "AAA###",   "color": (240, 240, 240)},
    "SD": {"name": "South Dakota",  "pattern": "######",   "color": (240, 240, 240)},
    "TN": {"name": "Tennessee",     "pattern": "AAA####",  "color": (240, 240, 240)},
    "TX": {"name": "Texas",         "pattern": "AAA####",  "color": (240, 240, 240)},
    "UT": {"name": "Utah",          "pattern": "A###BB",   "color": (240, 240, 240)},
    "VT": {"name": "Vermont",       "pattern": "AAA###",   "color": (240, 240, 240)},
    "VA": {"name": "Virginia",      "pattern": "AAA####",  "color": (240, 240, 240)},
    "WA": {"name": "Washington",    "pattern": "AAA####",  "color": (240, 240, 240)},
    "WV": {"name": "West Virginia", "pattern": "AAA####",  "color": (240, 240, 240)},
    "WI": {"name": "Wisconsin",     "pattern": "AAA####",  "color": (240, 240, 240)},
    "WY": {"name": "Wyoming",       "pattern": "######",   "color": (240, 240, 240)},
    "DC": {"name": "Washington DC", "pattern": "AA####",   "color": (240, 240, 240)},
}

AU_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
CA_PROVINCES = ["ON", "BC", "QC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]
IN_STATES = ["MH", "DL", "KA", "TN", "GJ", "RJ", "UP", "WB", "AP", "TS"]
JP_REGIONS = ["品川", "名古屋", "大阪", "札幌", "福岡", "横浜", "神戸", "京都"]


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_plate_text(pattern: str) -> str:
    """
    Generate plate text from a pattern:
        A = random uppercase letter (no I, O to avoid confusion)
        # = random digit
        B = random letter or digit
        - = literal hyphen
        any other char = literal
    """
    letters = [c for c in string.ascii_uppercase if c not in "IO"]
    result = []
    for ch in pattern:
        if ch == "A":
            result.append(random.choice(letters))
        elif ch == "#":
            result.append(str(random.randint(0, 9)))
        elif ch == "B":
            pool = letters + list("0123456789")
            result.append(random.choice(pool))
        else:
            result.append(ch)
    return "".join(result)


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

class PlateRenderer:
    def __init__(self, image_w: int = 640, image_h: int = 480):
        self.image_w = image_w
        self.image_h = image_h

    def render(self, fmt: PlateFormat, text: str,
               state_text: str = "") -> tuple[Image.Image, tuple]:
        """
        Render a plate onto a realistic background.
        Returns (PIL Image, bbox) where bbox = (x1, y1, x2, y2) in pixels.
        """
        # Create background scene
        img = self._make_background()

        # Render plate image
        plate_img = self._render_plate(fmt, text, state_text)

        # Random plate placement (not always centered)
        px = random.randint(
            max(10, self.image_w // 2 - plate_img.width),
            min(self.image_w - plate_img.width - 10,
                self.image_w // 2 + plate_img.width // 2)
        )
        py = random.randint(
            max(10, self.image_h // 2 - plate_img.height),
            min(self.image_h - plate_img.height - 10,
                self.image_h // 2 + plate_img.height // 2)
        )

        # Slight random rotation (-4 to +4 degrees)
        angle = random.uniform(-4, 4)
        plate_rotated = plate_img.rotate(angle, expand=True,
                                          fillcolor=self._bg_color())
        img.paste(plate_rotated, (px, py))

        x1, y1 = px, py
        x2, y2 = px + plate_rotated.width, py + plate_rotated.height

        return img, (x1, y1, x2, y2)

    def _bg_color(self):
        r = random.randint(60, 200)
        g = random.randint(60, 200)
        b = random.randint(60, 200)
        return (r, g, b)

    def _make_background(self) -> Image.Image:
        """Generate a plausible car/road background."""
        style = random.choice(["solid", "gradient", "road", "car_body"])
        img = Image.new("RGB", (self.image_w, self.image_h))
        draw = ImageDraw.Draw(img)

        if style == "solid":
            r = random.randint(30, 180)
            g = random.randint(30, 180)
            b = random.randint(30, 180)
            draw.rectangle([0, 0, self.image_w, self.image_h], fill=(r, g, b))

        elif style == "gradient":
            top = (random.randint(40, 120),) * 3
            bot = (random.randint(80, 200),) * 3
            for y in range(self.image_h):
                t = y / self.image_h
                c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
                draw.line([(0, y), (self.image_w, y)], fill=c)

        elif style == "road":
            # Grey road with white line markings
            draw.rectangle([0, 0, self.image_w, self.image_h], fill=(80, 80, 80))
            for x in range(0, self.image_w, 60):
                draw.rectangle([x, self.image_h // 2 - 4,
                                 x + 30, self.image_h // 2 + 4],
                                fill=(220, 220, 180))

        else:  # car_body
            body_color = (
                random.randint(30, 220),
                random.randint(30, 220),
                random.randint(30, 220),
            )
            draw.rectangle([0, 0, self.image_w, self.image_h], fill=body_color)

        # Optional: add noise grain
        if random.random() < 0.4:
            arr = np.array(img, dtype=np.int16)
            noise = np.random.randint(-12, 12, arr.shape, dtype=np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def _render_plate(self, fmt: PlateFormat, text: str,
                      state_text: str = "") -> Image.Image:
        """Draw the plate itself."""
        badge_w = 48 if fmt.has_country_badge else 0
        plate_w = fmt.width + badge_w
        plate_h = fmt.height

        plate = Image.new("RGB", (plate_w, plate_h), fmt.bg_color)
        draw = ImageDraw.Draw(plate)

        # Outer border
        draw.rectangle(
            [0, 0, plate_w - 1, plate_h - 1],
            outline=fmt.border_color, width=fmt.border_width
        )

        # EU-style country badge on left
        if fmt.has_country_badge:
            draw.rectangle([0, 0, badge_w, plate_h],
                           fill=fmt.badge_color)
            badge_font = find_font(18, bold=True)
            draw.text(
                (badge_w // 2, plate_h // 2),
                fmt.badge_text, fill=fmt.badge_text_color,
                font=badge_font, anchor="mm"
            )
            # Gold EU stars ring (simplified as dots)
            cx, cy = badge_w // 2, plate_h // 4
            for i in range(8):
                angle_r = math.radians(i * 45)
                sx = int(cx + 10 * math.cos(angle_r))
                sy = int(cy + 10 * math.sin(angle_r))
                draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2],
                             fill=(255, 204, 0))

        # Main text area starts after badge
        text_area_x = badge_w + fmt.border_width + 4
        text_area_w = plate_w - badge_w - fmt.border_width * 2 - 8

        main_font = find_font(fmt.font_size, bold=True)
        small_font = find_font(max(16, fmt.font_size // 4))

        if fmt.two_line and state_text:
            # State/region text on top, plate number below
            state_y = plate_h * 0.28
            plate_y = plate_h * 0.65
            draw.text(
                (text_area_x + text_area_w // 2, state_y),
                state_text, fill=fmt.text_color,
                font=small_font, anchor="mm"
            )
            draw.text(
                (text_area_x + text_area_w // 2, plate_y),
                text, fill=fmt.text_color,
                font=main_font, anchor="mm"
            )
        else:
            # Centered single line
            draw.text(
                (text_area_x + text_area_w // 2, plate_h // 2),
                text, fill=fmt.text_color,
                font=main_font, anchor="mm"
            )

        # Subtle inner shadow / highlight on plate edges
        if random.random() < 0.5:
            inset = fmt.border_width + 2
            draw.rectangle(
                [inset, inset, plate_w - inset, plate_h - inset],
                outline=(min(255, fmt.border_color[0] + 40),) * 3,
                width=1
            )

        return plate


# ---------------------------------------------------------------------------
# Augmentation pass
# ---------------------------------------------------------------------------

def augment_image(img: Image.Image) -> Image.Image:
    """Apply realistic capture-condition augmentations."""
    # Brightness
    factor = random.uniform(0.5, 1.5)
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Gaussian blur (simulate motion / defocus)
    if random.random() < 0.3:
        radius = random.uniform(0.5, 2.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Gaussian noise
    if random.random() < 0.4:
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, random.uniform(3, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Night mode
    if random.random() < 0.15:
        arr = (np.array(img, dtype=np.float32) * 0.25).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        # Add headlight bloom
        for _ in range(random.randint(0, 2)):
            cx = random.randint(0, img.width)
            cy = random.randint(0, img.height)
            r = random.randint(20, 60)
            for dr in range(r, 0, -5):
                alpha = int(180 * (1 - dr / r))
                draw.ellipse([cx - dr, cy - dr, cx + dr, cy + dr],
                             fill=(alpha, alpha, alpha))

    return img


# ---------------------------------------------------------------------------
# YOLO annotation writer
# ---------------------------------------------------------------------------

def to_yolo(bbox_px: tuple, img_w: int, img_h: int) -> str:
    """Convert pixel bbox (x1,y1,x2,y2) to YOLO normalised format."""
    x1, y1, x2, y2 = bbox_px
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------------------
# Region selectors
# ---------------------------------------------------------------------------

REGION_MAP = {
    "us": ["US_standard"],
    "eu": ["UK_front", "UK_rear", "DE", "FR", "EU_generic"],
    "asia": ["JP", "IN_front", "IN_rear"],
    "latam": ["BR"],
    "au": ["AU"],
    "ca": ["CA"],
}


def pick_format_and_state(region: str,
                           state_filter: Optional[str] = None
                           ) -> tuple[PlateFormat, str, str]:
    """
    Returns (PlateFormat, plate_text, state_text) for a given region.
    """
    fmt_keys = REGION_MAP.get(region, ["US_standard"])
    fmt_key = random.choice(fmt_keys)
    fmt = PLATE_FORMATS[fmt_key]

    state_text = fmt.state_text  # default from format

    if region == "us":
        if state_filter:
            state_code = state_filter.upper()
        else:
            state_code = random.choice(list(US_STATES.keys()))
        state_info = US_STATES[state_code]
        pattern = state_info["pattern"]
        fmt = PlateFormat(
            country="US", width=440, height=220,
            bg_color=state_info["color"], text_color=(20, 20, 20),
            border_color=(150, 150, 150), border_width=6,
            font_size=90, text_pattern=pattern,
            two_line=True, state_text=state_code,
        )
        state_text = state_code
        text = generate_plate_text(pattern)

    elif region == "au":
        state_text = random.choice(AU_STATES)
        fmt.state_text = state_text
        text = generate_plate_text(fmt.text_pattern)

    elif region == "ca":
        state_text = random.choice(CA_PROVINCES)
        fmt.state_text = state_text
        text = generate_plate_text(fmt.text_pattern)

    elif fmt_key in ("IN_front", "IN_rear"):
        in_state = random.choice(IN_STATES)
        num = random.randint(1, 99)
        state_text = f"{in_state} {num:02d}"
        fmt.state_text = state_text
        text = generate_plate_text(fmt.text_pattern)

    elif fmt_key == "JP":
        state_text = random.choice(JP_REGIONS)
        fmt.state_text = state_text
        text = generate_plate_text(fmt.text_pattern)

    else:
        text = generate_plate_text(fmt.text_pattern)

    return fmt, text, state_text


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    regions: list[str],
    count: int,
    output_dir: Path,
    augment: bool = True,
    image_w: int = 640,
    image_h: int = 480,
    state_filter: Optional[str] = None,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    renderer = PlateRenderer(image_w=image_w, image_h=image_h)
    manifest = []

    print(f"Generating {count} synthetic plates → {output_dir}")
    print(f"Regions: {regions} | Augment: {augment}")

    for i in range(count):
        region = random.choice(regions)
        fmt, text, state_text = pick_format_and_state(region, state_filter)

        img, bbox_px = renderer.render(fmt, text, state_text)

        if augment:
            img = augment_image(img)

        stem = f"syn_{region}_{i:06d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        img.save(str(img_path), "JPEG", quality=92)

        yolo_line = to_yolo(bbox_px, image_w, image_h)
        with open(lbl_path, "w") as f:
            # YOLO line + plate text as extra field for OCR training
            f.write(yolo_line + f" {text}\n")

        manifest.append({
            "image": str(img_path),
            "label_file": str(lbl_path),
            "text": text,
            "state_or_region": state_text,
            "country": fmt.country,
            "bbox_px": list(bbox_px),
            "yolo": yolo_line,
            "augmented": augment,
            "synthetic": True,
        })

        if (i + 1) % 500 == 0 or (i + 1) == count:
            print(f"  {i+1}/{count} images generated...")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Stats
    by_country = {}
    for m in manifest:
        by_country[m["country"]] = by_country.get(m["country"], 0) + 1

    stats = {
        "total": count,
        "by_country": by_country,
        "augmented": augment,
        "image_size": f"{image_w}x{image_h}",
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! {count} images saved to {output_dir}")
    print(f"Country breakdown:")
    for country, n in sorted(by_country.items(), key=lambda x: -x[1]):
        print(f"  {country}: {n}")
    print(f"\nNext step:")
    print(f"  python scripts/prepare_dataset.py --regions {' '.join(regions)} --split 80/10/10")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic licence plate images")
    p.add_argument("--count", type=int, default=2000,
                   help="Number of images to generate (default: 2000)")
    p.add_argument("--regions", nargs="+", default=["us"],
                   choices=["us", "eu", "asia", "latam", "au", "ca", "all"],
                   help="Which plate formats to include")
    p.add_argument("--output", default="data/raw/synthetic",
                   help="Output directory (default: data/raw/synthetic)")
    p.add_argument("--augment", action="store_true", default=True,
                   help="Apply augmentations (brightness, blur, noise, night mode)")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable augmentations")
    p.add_argument("--image-width",  type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--state", default=None,
                   help="US only: fix to a specific state code e.g. --state CA")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    regions = (
        list(REGION_MAP.keys())
        if "all" in args.regions
        else args.regions
    )

    output_dir = Path(args.output)

    generate_dataset(
        regions=regions,
        count=args.count,
        output_dir=output_dir,
        augment=args.augment,
        image_w=args.image_width,
        image_h=args.image_height,
        state_filter=args.state,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

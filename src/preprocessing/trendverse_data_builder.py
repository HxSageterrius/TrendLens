"""
TrendLens — Fashion Trends Dataset Builder
Builds structured fashion dataset inspired by TrendVerse Instagram & Viral Trends schema.
Generates:
1. Time-series trend dataset for forecasting (weekly engagement / search volume over 2 years)
2. Fashion post metadata (item_id, category, style, likes, tags, timestamps)
3. High-quality fashion sample images for visual feature extraction and similarity search
"""

import math
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def create_fashion_image(
    output_path: Path,
    category: str,
    style: str,
    primary_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    pattern: str = "solid",
    size: Tuple[int, int] = (300, 300)
):
    """Generates a structured fashion garment image with distinct visual features."""
    w, h = size
    # Modern studio background gradient / light neutral
    bg_color = (240, 242, 245)
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Subtle studio vignette / backdrop
    for i in range(15):
        draw.rectangle([i, i, w - i, h - i], outline=(232 + i, 235 + i, 238 + i))

    cx, cy = w // 2, h // 2

    # Draw garment based on category
    if category in ["dress", "gown"]:
        # Dress geometry: bodice + flared skirt
        bodice = [(cx - 45, cy - 80), (cx + 45, cy - 80), (cx + 35, cy - 10), (cx - 35, cy - 10)]
        skirt = [(cx - 35, cy - 10), (cx + 35, cy - 10), (cx + 85, cy + 95), (cx - 85, cy + 95)]
        draw.polygon(bodice, fill=primary_color, outline=accent_color)
        draw.polygon(skirt, fill=primary_color, outline=accent_color)
        # Neckline
        draw.arc([cx - 25, cy - 95, cx + 25, cy - 65], 0, 180, fill=accent_color, width=3)
        # Waist belt/accent
        draw.line([(cx - 38, cy - 10), (cx + 38, cy - 10)], fill=accent_color, width=4)

    elif category in ["jacket", "coat", "outerwear"]:
        # Outerwear geometry: boxy torso + sleeves + lapel
        torso = [(cx - 65, cy - 75), (cx + 65, cy - 75), (cx + 60, cy + 80), (cx - 60, cy + 80)]
        draw.polygon(torso, fill=primary_color, outline=accent_color)
        # Sleeves
        draw.polygon([(cx - 65, cy - 70), (cx - 95, cy + 40), (cx - 75, cy + 45), (cx - 50, cy - 40)], fill=primary_color, outline=accent_color)
        draw.polygon([(cx + 65, cy - 70), (cx + 95, cy + 40), (cx + 75, cy + 45), (cx + 50, cy - 40)], fill=primary_color, outline=accent_color)
        # Lapels / zipper
        draw.line([(cx, cy - 75), (cx, cy + 80)], fill=accent_color, width=4)
        draw.polygon([(cx - 25, cy - 75), (cx, cy - 35), (cx, cy - 75)], fill=accent_color)
        draw.polygon([(cx + 25, cy - 75), (cx, cy - 35), (cx, cy - 75)], fill=accent_color)

    elif category in ["shirt", "top", "hoodie"]:
        # Shirt geometry: shoulders + sleeves + collar
        torso = [(cx - 55, cy - 65), (cx + 55, cy - 65), (cx + 50, cy + 70), (cx - 50, cy + 70)]
        draw.polygon(torso, fill=primary_color, outline=accent_color)
        # Short / Medium sleeves
        draw.polygon([(cx - 55, cy - 60), (cx - 85, cy + 5), (cx - 65, cy + 10), (cx - 45, cy - 30)], fill=primary_color, outline=accent_color)
        draw.polygon([(cx + 55, cy - 60), (cx + 85, cy + 5), (cx + 65, cy + 10), (cx + 45, cy - 30)], fill=primary_color, outline=accent_color)
        # Collar / Graphic
        draw.ellipse([cx - 20, cy - 75, cx + 20, cy - 50], outline=accent_color, width=3)
        if pattern == "graphic":
            draw.rectangle([cx - 25, cy - 20, cx + 25, cy + 20], fill=accent_color)

    elif category in ["trousers", "jeans", "pants"]:
        # Trousers geometry: waist + two legs
        waist = [(cx - 50, cy - 80), (cx + 50, cy - 80), (cx + 48, cy - 50), (cx - 48, cy - 50)]
        leg_left = [(cx - 48, cy - 50), (cx - 5, cy - 40), (cx - 15, cy + 95), (cx - 55, cy + 95)]
        leg_right = [(cx + 5, cy - 40), (cx + 48, cy - 50), (cx + 55, cy + 95), (cx + 15, cy + 95)]
        draw.polygon(waist, fill=primary_color, outline=accent_color)
        draw.polygon(leg_left, fill=primary_color, outline=accent_color)
        draw.polygon(leg_right, fill=primary_color, outline=accent_color)
        # Pocket seams
        draw.arc([cx - 45, cy - 65, cx - 20, cy - 40], 0, 90, fill=accent_color, width=2)
        draw.arc([cx + 20, cy - 65, cx + 45, cy - 40], 90, 180, fill=accent_color, width=2)

    else:
        # Skirt / default
        skirt = [(cx - 40, cy - 50), (cx + 40, cy - 50), (cx + 75, cy + 70), (cx - 75, cy + 70)]
        draw.polygon(skirt, fill=primary_color, outline=accent_color)
        draw.line([(cx - 40, cy - 50), (cx + 40, cy - 50)], fill=accent_color, width=5)

    # Patterns
    if pattern == "striped":
        for sy in range(cy - 60, cy + 60, 18):
            draw.line([(cx - 40, sy), (cx + 40, sy)], fill=accent_color, width=2)
    elif pattern == "dots":
        for dx in [-25, 0, 25]:
            for dy in [-25, 0, 25]:
                draw.ellipse([cx + dx - 4, cy + dy - 4, cx + dx + 4, cy + dy + 4], fill=accent_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=92)


def build_fashion_dataset(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the complete fashion catalog and time-series trends datasets.
    """
    trends_dir = base_dir / "data" / "fashion_trends"
    images_dir = trends_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Catalog of Fashion Items (Styles & Categories)
    fashion_templates = [
        # Streetwear
        ("Oversized Cyber Hoodie", "shirt", "Streetwear", (45, 52, 60), (0, 210, 255), "graphic", 3420),
        ("Cargo Techwear Pants", "trousers", "Streetwear", (35, 40, 45), (200, 255, 0), "solid", 2890),
        ("Graffiti Puffer Jacket", "jacket", "Streetwear", (30, 30, 35), (255, 60, 90), "graphic", 4120),
        ("Urban Utility Vest", "jacket", "Streetwear", (70, 75, 65), (255, 180, 0), "solid", 1950),
        ("Retro Graphic Tee", "shirt", "Streetwear", (220, 225, 230), (240, 40, 80), "graphic", 2780),
        ("Baggy Skate Chinos", "trousers", "Streetwear", (180, 150, 110), (60, 50, 40), "solid", 2150),

        # Y2K Aesthetic
        ("Silver Metallic Crop Top", "shirt", "Y2K", (210, 215, 225), (255, 105, 180), "solid", 5210),
        ("Low-Rise Flare Jeans", "trousers", "Y2K", (65, 115, 175), (255, 182, 193), "solid", 4680),
        ("Pink Velour Track Jacket", "jacket", "Y2K", (235, 100, 160), (255, 255, 255), "striped", 6100),
        ("Glitter Mesh Bodycon Dress", "dress", "Y2K", (160, 90, 190), (0, 240, 255), "dots", 5830),
        ("Butterfly Halter Top", "shirt", "Y2K", (120, 220, 230), (255, 90, 150), "graphic", 4320),
        ("Pleated Denim Mini Skirt", "skirt", "Y2K", (75, 125, 185), (240, 240, 250), "solid", 3990),

        # Minimalist & Quiet Luxury
        ("Cashmere Crewneck Sweater", "shirt", "Minimalist", (225, 220, 210), (140, 130, 120), "solid", 3800),
        ("Tailored Wide-Leg Trousers", "trousers", "Minimalist", (50, 50, 55), (180, 180, 185), "solid", 4250),
        ("Structure Wool Trench Coat", "jacket", "Minimalist", (195, 165, 130), (70, 55, 40), "solid", 5400),
        ("Silk Slip Maxi Dress", "dress", "Minimalist", (30, 32, 35), (210, 205, 195), "solid", 4890),
        ("Oatmeal Linen Button-Up", "shirt", "Minimalist", (240, 235, 225), (160, 150, 140), "solid", 3120),
        ("Straight Wool Slacks", "trousers", "Minimalist", (80, 85, 90), (200, 200, 200), "solid", 2980),

        # Denim Core
        ("Raw Selvedge Denim Jacket", "jacket", "Denim", (30, 60, 110), (190, 140, 70), "solid", 3740),
        ("Vintage High-Waist Mom Jeans", "trousers", "Denim", (90, 145, 200), (240, 240, 240), "solid", 4120),
        ("Distressed Trucker Jacket", "jacket", "Denim", (110, 160, 210), (60, 80, 110), "striped", 3450),
        ("Denim Western Workshirt", "shirt", "Denim", (70, 120, 170), (220, 220, 230), "solid", 2670),
        ("Carpenter Denim Jeans", "trousers", "Denim", (50, 85, 140), (210, 180, 130), "solid", 3210),
        ("Tiered Chambray Denim Dress", "dress", "Denim", (100, 150, 195), (40, 60, 90), "solid", 3560),

        # Cottagecore & Bohemian
        ("Floral Prairie Midi Dress", "dress", "Cottagecore", (245, 230, 215), (180, 70, 85), "dots", 4520),
        ("Puff-Sleeve Linen Blouse", "shirt", "Cottagecore", (250, 245, 235), (140, 175, 130), "solid", 3890),
        ("Knit Cable Cardigan", "jacket", "Cottagecore", (215, 195, 165), (120, 90, 60), "striped", 4150),
        ("Embroidered A-Line Skirt", "skirt", "Cottagecore", (120, 145, 120), (240, 200, 160), "dots", 3200),
        ("Corset Lace-Up Bodice Top", "shirt", "Cottagecore", (200, 160, 140), (100, 60, 50), "solid", 4410),
        ("Tiered Ruffle Smock Dress", "dress", "Cottagecore", (230, 210, 200), (90, 130, 110), "dots", 3780),

        # Athleisure & Sportcore
        ("Seamless Compression Top", "shirt", "Athleisure", (40, 42, 48), (0, 230, 180), "solid", 3650),
        ("High-Waist Sculpt Leggings", "trousers", "Athleisure", (30, 30, 35), (255, 95, 110), "solid", 4890),
        ("Windbreaker Track Jacket", "jacket", "Athleisure", (25, 35, 60), (255, 190, 0), "striped", 3910),
        ("Cropped Zip Running Hoodie", "jacket", "Athleisure", (150, 160, 170), (0, 180, 255), "solid", 3140),
        ("Ribbed Bike Shorts", "trousers", "Athleisure", (60, 65, 75), (200, 255, 100), "solid", 2800),
        ("Athletic Tennis Dress", "dress", "Athleisure", (245, 245, 245), (40, 120, 80), "solid", 4210),
    ]

    post_records = []
    for idx, (title, cat, style, p_col, a_col, pat, base_likes) in enumerate(fashion_templates, start=1):
        item_id = f"FSH_{idx:04d}"
        filename = f"{item_id}_{style.lower()}_{cat}.jpg"
        img_path = images_dir / filename
        rel_img_path = f"data/fashion_trends/images/{filename}"

        # Create image
        create_fashion_image(
            output_path=img_path,
            category=cat,
            style=style,
            primary_color=p_col,
            accent_color=a_col,
            pattern=pat
        )

        # Engagement stats & realistic timestamps
        # Spread across past 18 months
        days_ago = random.randint(10, 500)
        timestamp = (pd.Timestamp("2026-06-01") - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
        likes = int(base_likes * random.uniform(0.85, 1.25))
        comments = int(likes * random.uniform(0.04, 0.08))

        post_records.append({
            "item_id": item_id,
            "title": title,
            "category": cat,
            "style": style,
            "pattern": pat,
            "likes": likes,
            "comments": comments,
            "timestamp": timestamp,
            "file_path": rel_img_path,
            "tags": f"#{style.lower()} #{cat} #fashiontrend #ootd #styleinspo"
        })

    posts_df = pd.DataFrame(post_records)
    posts_csv_path = trends_dir / "fashion_posts_metadata.csv"
    posts_df.to_csv(posts_csv_path, index=False)

    # 2. Time-Series Trends Dataset (Weekly volume over 2 years: 2024-07 to 2026-06)
    # Styles: Streetwear, Y2K, Minimalist, Denim, Cottagecore, Athleisure
    dates = pd.date_range(start="2024-07-07", end="2026-06-07", freq="W-SUN")
    trend_rows = []

    # Dynamic trajectories:
    # Y2K: strong surge in 2024-2025, now stabilizing/peaking
    # Minimalist: steady consistent growth (Quiet Luxury)
    # Streetwear: high volume with cyclical waves
    # Denim: timeless base with seasonal spring/autumn spikes
    # Cottagecore: summer seasonal peaks
    # Athleisure: surging steadily with high health/active trend
    for i, dt in enumerate(dates):
        week_num = i
        month = dt.month

        # Base time trends with seasonality + noise
        noise = lambda: np.random.normal(0, 2.5)

        # Y2K: peak around week 60-70
        y2k_val = max(20, 50 + 40 * math.sin((week_num / 52.0) * math.pi) + (week_num * 0.3) + noise())

        # Minimalist: upward linear slope
        min_val = max(25, 45 + (week_num * 0.65) + 8 * math.cos((month / 12.0) * 2 * math.pi) + noise())

        # Streetwear: large cyclical wave
        street_val = max(35, 70 + 15 * math.sin((week_num / 26.0) * math.pi) + noise())

        # Denim: baseline with fall/spring surges (months 3-4 and 9-10)
        spring_fall = 18 if month in [3, 4, 9, 10] else 0
        denim_val = max(30, 55 + spring_fall + (week_num * 0.15) + noise())

        # Cottagecore: high in summer (months 5, 6, 7)
        summer_boost = 35 if month in [5, 6, 7] else (-15 if month in [11, 12, 1] else 0)
        cottage_val = max(15, 40 + summer_boost + noise())

        # Athleisure: strong upward slope + January resolution surge
        jan_boost = 25 if month == 1 else 0
        ath_val = max(30, 50 + (week_num * 0.55) + jan_boost + noise())

        for style, val, hashtag in [
            ("Y2K", y2k_val, "#y2kfashion"),
            ("Minimalist", min_val, "#quietluxury"),
            ("Streetwear", street_val, "#streetwear"),
            ("Denim", denim_val, "#denimoutfit"),
            ("Cottagecore", cottage_val, "#cottagecore"),
            ("Athleisure", ath_val, "#athleisure"),
        ]:
            trend_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "style": style,
                "hashtag": hashtag,
                "trend_score": round(float(val), 2),
                "search_volume": int(val * 420 + random.randint(100, 500)),
                "social_engagement": int(val * 310 + random.randint(50, 300)),
            })

    trends_df = pd.DataFrame(trend_rows)
    trends_csv_path = trends_dir / "insta_fashion_trends.csv"
    trends_df.to_csv(trends_csv_path, index=False)

    print(f"Generated {len(posts_df)} fashion items in {posts_csv_path}")
    print(f"Generated {len(trends_df)} weekly trend datapoints in {trends_csv_path}")

    return posts_df, trends_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    build_fashion_dataset(project_root)

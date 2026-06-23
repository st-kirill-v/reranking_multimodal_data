from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(path, size)


GREEN = "#007934"
DARK = "#1c2731"
MUTED = "#5b6977"
BLUE = "#1a569c"
ORANGE = "#de7626"
GREY = "#dfe5e8"
LIGHT = "#f6f8f7"
WHITE = "#ffffff"


def text(draw, xy, value, size=22, fill=DARK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def draw_bar_chart(path: Path, title: str, rows, max_value: float, x_label: str):
    w, h = 1200, 760
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 16], fill=GREEN)
    text(d, (60, 48), title, 36, DARK, True)
    text(d, (60, 94), x_label, 20, MUTED)

    left, top = 290, 160
    bar_w, bar_h, gap = 760, 54, 38
    for i, (label, value, color) in enumerate(rows):
        y = top + i * (bar_h + gap)
        text(
            d,
            (60, y + 10),
            label,
            24,
            DARK,
            True if "Multimodal" in label or "Adaptive" in label else False,
        )
        d.rounded_rectangle([left, y, left + bar_w, y + bar_h], radius=8, fill=GREY)
        d.rounded_rectangle(
            [left, y, left + int(bar_w * value / max_value), y + bar_h], radius=8, fill=color
        )
        text(d, (left + bar_w + 28, y + 12), f"{value:.4f}", 24, DARK, True)

    d.line([left, h - 120, left + bar_w, h - 120], fill=MUTED, width=2)
    for tick in [0.5, 0.6, 0.7]:
        x = left + int(bar_w * tick / max_value)
        d.line([x, h - 127, x, h - 113], fill=MUTED, width=2)
        text(d, (x, h - 94), f"{tick:.1f}", 18, MUTED, anchor="mm")
    img.save(path)


def draw_route_distribution(path: Path):
    rows = [
        ("table_or_text", 203, GREEN),
        ("visual", 88, BLUE),
        ("high_confidence", 17, ORANGE),
    ]
    w, h = 1200, 760
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 16], fill=GREEN)
    text(d, (60, 48), "Adaptive reranking route distribution", 36, DARK, True)
    text(d, (60, 94), "308 questions, routing before reranking", 20, MUTED)
    total = sum(v for _, v, _ in rows)

    left, top = 330, 165
    bar_w, bar_h, gap = 690, 62, 52
    for i, (label, value, color) in enumerate(rows):
        y = top + i * (bar_h + gap)
        pct = value / total * 100
        text(d, (60, y + 16), label, 26, DARK, True)
        d.rounded_rectangle([left, y, left + bar_w, y + bar_h], radius=8, fill=GREY)
        d.rounded_rectangle(
            [left, y, left + int(bar_w * value / total), y + bar_h], radius=8, fill=color
        )
        text(d, (left + bar_w + 26, y + 14), f"{value} ({pct:.1f}%)", 24, DARK, True)

    cards = [
        ("Mean F1 by route", "table_or_text 0.7185\nvisual 0.6549\nhigh_confidence 0.7296", GREEN),
        (
            "Latency by route",
            "table_or_text 9.8231s\nvisual 11.0222s\nhigh_confidence 2.9837s",
            BLUE,
        ),
    ]
    for i, (head, body, color) in enumerate(cards):
        x = 70 + i * 530
        y = 565
        d.rounded_rectangle([x, y, x + 470, y + 135], radius=10, fill=LIGHT, outline=GREY, width=2)
        d.rectangle([x, y, x + 10, y + 135], fill=color)
        text(d, (x + 28, y + 18), head, 22, DARK, True)
        text(d, (x + 28, y + 52), body, 20, MUTED)
    img.save(path)


def draw_tradeoff(path: Path):
    points = [
        ("Text Reranker", 1.2344, 0.5497, MUTED),
        ("Fast Fusion", 2.5080, 0.6575, ORANGE),
        ("No Reranker", 3.4263, 0.6784, BLUE),
        ("Adaptive", 9.7882, 0.7009, GREEN),
        ("Multimodal", 13.6441, 0.7023, DARK),
    ]
    w, h = 1200, 760
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 16], fill=GREEN)
    text(d, (60, 48), "Quality-latency trade-off", 36, DARK, True)
    text(d, (60, 94), "Mean F1 vs end-to-end latency", 20, MUTED)

    left, top, plot_w, plot_h = 120, 150, 920, 470
    d.line([left, top + plot_h, left + plot_w, top + plot_h], fill=MUTED, width=3)
    d.line([left, top, left, top + plot_h], fill=MUTED, width=3)
    text(d, (left + plot_w - 35, top + plot_h + 48), "Latency, sec", 20, MUTED, anchor="mm")
    text(d, (left - 35, top - 25), "Mean F1", 20, MUTED, anchor="mm")

    xmin, xmax = 0.0, 15.0
    ymin, ymax = 0.50, 0.72
    for tick in [0, 5, 10, 15]:
        x = left + int((tick - xmin) / (xmax - xmin) * plot_w)
        d.line([x, top + plot_h, x, top + plot_h + 10], fill=MUTED, width=2)
        text(d, (x, top + plot_h + 30), str(tick), 16, MUTED, anchor="mm")
    for tick in [0.50, 0.60, 0.70]:
        y = top + plot_h - int((tick - ymin) / (ymax - ymin) * plot_h)
        d.line([left - 10, y, left, y], fill=MUTED, width=2)
        text(d, (left - 35, y), f"{tick:.2f}", 16, MUTED, anchor="mm")
        d.line([left, y, left + plot_w, y], fill="#edf1f3", width=1)

    for label, latency, f1, color in points:
        x = left + int((latency - xmin) / (xmax - xmin) * plot_w)
        y = top + plot_h - int((f1 - ymin) / (ymax - ymin) * plot_h)
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=color)
        if label == "Adaptive":
            text(d, (x - 70, y - 42), label, 18, DARK, True)
        elif label == "Multimodal":
            text(d, (x - 82, y + 22), label, 18, DARK, True)
        else:
            text(d, (x + 14, y - 13), label, 18, DARK, False)

    d.rounded_rectangle([92, 660, 1085, 720], radius=10, fill=LIGHT, outline=GREY, width=2)
    text(
        d,
        (115, 675),
        "Adaptive Reranking: Mean F1 0.7009 vs 0.7023 for best multimodal;",
        19,
        GREEN,
        True,
    )
    text(d, (115, 700), "latency 9.7882s vs 13.6441s, approximately 28% lower.", 19, GREEN, True)
    img.save(path)


draw_bar_chart(
    OUT / "reranking_quality_comparison.png",
    "Answer quality comparison across reranking strategies",
    [
        ("No Reranker", 0.6784, BLUE),
        ("Text Reranker", 0.5497, MUTED),
        ("Multimodal Reranker", 0.7023, GREEN),
        ("Adaptive Reranking", 0.7009, ORANGE),
    ],
    0.75,
    "Mean F1",
)
draw_route_distribution(OUT / "adaptive_route_distribution.png")
draw_tradeoff(OUT / "quality_latency_tradeoff.png")

print("generated figures in reports/figures")

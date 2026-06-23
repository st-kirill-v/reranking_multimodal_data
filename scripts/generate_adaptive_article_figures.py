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
GRID = "#e9eef1"
WHITE = "#ffffff"


def text(draw, xy, value, size=22, fill=DARK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def svg_text(x, y, value, size=18, fill=DARK, weight="400", anchor="start"):
    escaped = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escaped}</text>'
    )


def save_svg(path: Path, width: int, height: int, body: list[str]):
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def draw_scatter_quality_latency():
    png = OUT / "reranking_quality_latency_scatter.png"
    svg = OUT / "reranking_quality_latency_scatter.svg"
    points = [
        ("Text Reranker", 1.2344, 0.5497, MUTED),
        ("No Reranker", 3.4263, 0.6784, BLUE),
        ("Adaptive Reranking", 9.7882, 0.7009, ORANGE),
        ("Multimodal Reranker", 13.6441, 0.7023, GREEN),
    ]
    w, h = 1800, 1150
    left, top, plot_w, plot_h = 190, 190, 1380, 760
    xmin, xmax = 0.0, 15.0
    ymin, ymax = 0.50, 0.73

    def sx(x):
        return left + int((x - xmin) / (xmax - xmin) * plot_w)

    def sy(y):
        return top + plot_h - int((y - ymin) / (ymax - ymin) * plot_h)

    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    text(
        d,
        (w // 2, 70),
        "Сравнение качества и задержки различных стратегий реранкинга",
        40,
        DARK,
        True,
        "mm",
    )
    text(d, (w // 2, 122), "Средняя задержка (секунды) / Mean F1", 26, MUTED, False, "mm")

    body = [
        svg_text(
            w // 2,
            70,
            "Сравнение качества и задержки различных стратегий реранкинга",
            40,
            DARK,
            "700",
            "middle",
        ),
        svg_text(w // 2, 122, "Средняя задержка (секунды) / Mean F1", 26, MUTED, "400", "middle"),
    ]

    for tick in [0, 5, 10, 15]:
        x = sx(tick)
        d.line([x, top, x, top + plot_h], fill=GRID, width=2)
        d.line([x, top + plot_h, x, top + plot_h + 12], fill=MUTED, width=2)
        text(d, (x, top + plot_h + 48), str(tick), 22, MUTED, anchor="mm")
        body.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x}" y1="{top + plot_h}" x2="{x}" y2="{top + plot_h + 12}" stroke="{MUTED}" stroke-width="2"/>'
        )
        body.append(svg_text(x, top + plot_h + 48, tick, 22, MUTED, "400", "middle"))

    for tick in [0.50, 0.55, 0.60, 0.65, 0.70]:
        y = sy(tick)
        d.line([left, y, left + plot_w, y], fill=GRID, width=2)
        d.line([left - 12, y, left, y], fill=MUTED, width=2)
        text(d, (left - 45, y), f"{tick:.2f}", 22, MUTED, anchor="mm")
        body.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="{GRID}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{left - 12}" y1="{y}" x2="{left}" y2="{y}" stroke="{MUTED}" stroke-width="2"/>'
        )
        body.append(svg_text(left - 45, y + 7, f"{tick:.2f}", 22, MUTED, "400", "middle"))

    d.line([left, top + plot_h, left + plot_w, top + plot_h], fill=DARK, width=3)
    d.line([left, top, left, top + plot_h], fill=DARK, width=3)
    text(d, (left + plot_w // 2, h - 78), "Средняя задержка (секунды)", 27, DARK, True, "mm")
    text(d, (70, top + plot_h // 2), "Mean F1", 27, DARK, True, "mm")
    body.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{DARK}" stroke-width="3"/>'
    )
    body.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{DARK}" stroke-width="3"/>'
    )
    body.append(
        svg_text(
            left + plot_w // 2, h - 78, "Средняя задержка (секунды)", 27, DARK, "700", "middle"
        )
    )
    body.append(svg_text(70, top + plot_h // 2, "Mean F1", 27, DARK, "700", "middle"))

    label_offsets = {
        "Text Reranker": (25, -22),
        "No Reranker": (25, -22),
        "Adaptive Reranking": (-165, -32),
        "Multimodal Reranker": (-220, 45),
    }
    for label, latency, f1, color in points:
        x, y = sx(latency), sy(f1)
        d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=color)
        dx, dy = label_offsets[label]
        text(d, (x + dx, y + dy), label, 25, DARK, True)
        body.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{color}"/>')
        body.append(svg_text(x + dx, y + dy + 22, label, 25, DARK, "700"))

    img.save(png, dpi=(300, 300))
    save_svg(svg, w, h, body)


def draw_adaptive_comparison():
    png = OUT / "adaptive_quality_latency_barchart.png"
    svg = OUT / "adaptive_quality_latency_barchart.svg"
    w, h = 1800, 1100
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    text(
        d,
        (w // 2, 70),
        "Влияние Adaptive Reranking на компромисс между качеством и задержкой",
        40,
        DARK,
        True,
        "mm",
    )
    body = [
        svg_text(
            w // 2,
            70,
            "Влияние Adaptive Reranking на компромисс между качеством и задержкой",
            40,
            DARK,
            "700",
            "middle",
        )
    ]

    panels = [
        (
            "Mean F1",
            [("Multimodal Reranker", 0.7023, GREEN), ("Adaptive Reranking", 0.7009, ORANGE)],
            0.75,
            150,
            0.001,
        ),
        (
            "Latency, seconds",
            [("Multimodal Reranker", 13.6441, GREEN), ("Adaptive Reranking", 9.7882, ORANGE)],
            15.0,
            970,
            0.01,
        ),
    ]
    for title, rows, maxv, x0, _ in panels:
        y0, plot_w, plot_h = 210, 610, 650
        text(d, (x0 + plot_w // 2, 160), title, 30, DARK, True, "mm")
        body.append(svg_text(x0 + plot_w // 2, 160, title, 30, DARK, "700", "middle"))
        d.line([x0, y0 + plot_h, x0 + plot_w, y0 + plot_h], fill=DARK, width=3)
        d.line([x0, y0, x0, y0 + plot_h], fill=DARK, width=3)
        body.append(
            f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="{DARK}" stroke-width="3"/>'
        )
        body.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="{DARK}" stroke-width="3"/>'
        )
        bar_w = 150
        for i, (label, value, color) in enumerate(rows):
            x = x0 + 135 + i * 250
            bh = int(plot_h * value / maxv)
            y = y0 + plot_h - bh
            d.rectangle([x, y, x + bar_w, y0 + plot_h], fill=color)
            text(
                d,
                (x + bar_w // 2, y - 28),
                f"{value:.4f}" if maxv < 2 else f"{value:.4f}s",
                23,
                DARK,
                True,
                "mm",
            )
            text(
                d,
                (x + bar_w // 2, y0 + plot_h + 45),
                label.replace(" ", "\n"),
                20,
                DARK,
                False,
                "mm",
            )
            body.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{color}"/>')
            body.append(
                svg_text(
                    x + bar_w // 2,
                    y - 28,
                    f"{value:.4f}" if maxv < 2 else f"{value:.4f}s",
                    23,
                    DARK,
                    "700",
                    "middle",
                )
            )
            for j, part in enumerate(label.split(" ")):
                body.append(
                    svg_text(
                        x + bar_w // 2, y0 + plot_h + 38 + j * 23, part, 20, DARK, "400", "middle"
                    )
                )

    img.save(png, dpi=(300, 300))
    save_svg(svg, w, h, body)


def draw_mean_f1_barplot():
    png = OUT / "reranking_mean_f1_barplot.png"
    svg = OUT / "reranking_mean_f1_barplot.svg"
    rows = [
        ("No Reranker", 0.6784, BLUE),
        ("Text Reranker", 0.5497, MUTED),
        ("Multimodal Reranker", 0.7023, GREEN),
        ("Adaptive Reranking", 0.7009, ORANGE),
    ]
    w, h = 1800, 1100
    left, top, plot_w, plot_h = 190, 190, 1400, 700
    ymax = 0.75
    img = Image.new("RGB", (w, h), WHITE)
    d = ImageDraw.Draw(img)
    text(
        d,
        (w // 2, 70),
        "Сравнение качества ответа различных стратегий реранкинга",
        40,
        DARK,
        True,
        "mm",
    )
    body = [
        svg_text(
            w // 2,
            70,
            "Сравнение качества ответа различных стратегий реранкинга",
            40,
            DARK,
            "700",
            "middle",
        )
    ]

    for tick in [0.0, 0.25, 0.50, 0.75]:
        y = top + plot_h - int(plot_h * tick / ymax)
        d.line([left, y, left + plot_w, y], fill=GRID, width=2)
        text(d, (left - 50, y), f"{tick:.2f}", 22, MUTED, anchor="mm")
        body.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="{GRID}" stroke-width="2"/>'
        )
        body.append(svg_text(left - 50, y + 7, f"{tick:.2f}", 22, MUTED, "400", "middle"))
    d.line([left, top + plot_h, left + plot_w, top + plot_h], fill=DARK, width=3)
    d.line([left, top, left, top + plot_h], fill=DARK, width=3)
    body.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{DARK}" stroke-width="3"/>'
    )
    body.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{DARK}" stroke-width="3"/>'
    )

    bar_w = 190
    gap = 140
    for i, (label, value, color) in enumerate(rows):
        x = left + 120 + i * (bar_w + gap)
        bh = int(plot_h * value / ymax)
        y = top + plot_h - bh
        d.rectangle([x, y, x + bar_w, top + plot_h], fill=color)
        text(d, (x + bar_w // 2, y - 28), f"{value:.4f}", 25, DARK, True, "mm")
        label_parts = label.split(" ")
        for j, part in enumerate(label_parts):
            text(d, (x + bar_w // 2, top + plot_h + 45 + j * 28), part, 22, DARK, False, "mm")
        body.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="{color}"/>')
        body.append(svg_text(x + bar_w // 2, y - 28, f"{value:.4f}", 25, DARK, "700", "middle"))
        for j, part in enumerate(label_parts):
            body.append(
                svg_text(
                    x + bar_w // 2, top + plot_h + 45 + j * 28, part, 22, DARK, "400", "middle"
                )
            )
    text(d, (75, top + plot_h // 2), "Mean F1", 27, DARK, True, "mm")
    body.append(svg_text(75, top + plot_h // 2, "Mean F1", 27, DARK, "700", "middle"))

    img.save(png, dpi=(300, 300))
    save_svg(svg, w, h, body)


draw_scatter_quality_latency()
draw_adaptive_comparison()
draw_mean_f1_barplot()
print("generated updated article figures in reports/figures")

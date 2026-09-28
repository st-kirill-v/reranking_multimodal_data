from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(path, size)


DARK = "#1c2731"
MUTED = "#5b6977"
BLUE = "#1a569c"
ORANGE = "#de7626"
GREEN = "#007934"
PURPLE = "#6D28D9"
GRID = "#e9eef1"
WHITE = "#ffffff"


def draw_text(draw, xy, value, size=22, fill=DARK, bold=False, anchor=None):
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


def main() -> None:
    png = OUT / "reranking_quality_latency_scatter_main.png"
    svg = OUT / "reranking_quality_latency_scatter_main.svg"
    points = [
        ("Text Reranker", 1.1160, 0.5674, ORANGE),
        ("Fusion", 2.5080, 0.6575, PURPLE),
        ("No Reranker", 3.4263, 0.6784, BLUE),
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
    draw = ImageDraw.Draw(img)
    body: list[str] = []

    title = "Сравнение качества и задержки различных стратегий реранкинга"
    subtitle = "Средняя задержка (секунды) / Mean F1"
    draw_text(draw, (w // 2, 70), title, 40, DARK, True, "mm")
    draw_text(draw, (w // 2, 122), subtitle, 26, MUTED, False, "mm")
    body.append(svg_text(w // 2, 70, title, 40, DARK, "700", "middle"))
    body.append(svg_text(w // 2, 122, subtitle, 26, MUTED, "400", "middle"))

    for tick in [0, 5, 10, 15]:
        x = sx(tick)
        draw.line([x, top, x, top + plot_h], fill=GRID, width=2)
        draw.line([x, top + plot_h, x, top + plot_h + 12], fill=MUTED, width=2)
        draw_text(draw, (x, top + plot_h + 48), str(tick), 22, MUTED, anchor="mm")
        body.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x}" y1="{top + plot_h}" x2="{x}" y2="{top + plot_h + 12}" stroke="{MUTED}" stroke-width="2"/>'
        )
        body.append(svg_text(x, top + plot_h + 48, tick, 22, MUTED, "400", "middle"))

    for tick in [0.50, 0.55, 0.60, 0.65, 0.70]:
        y = sy(tick)
        draw.line([left, y, left + plot_w, y], fill=GRID, width=2)
        draw.line([left - 12, y, left, y], fill=MUTED, width=2)
        draw_text(draw, (left - 45, y), f"{tick:.2f}", 22, MUTED, anchor="mm")
        body.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="{GRID}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{left - 12}" y1="{y}" x2="{left}" y2="{y}" stroke="{MUTED}" stroke-width="2"/>'
        )
        body.append(svg_text(left - 45, y + 7, f"{tick:.2f}", 22, MUTED, "400", "middle"))

    draw.line([left, top + plot_h, left + plot_w, top + plot_h], fill=DARK, width=3)
    draw.line([left, top, left, top + plot_h], fill=DARK, width=3)
    draw_text(
        draw, (left + plot_w // 2, h - 78), "Средняя задержка (секунды)", 27, DARK, True, "mm"
    )
    draw_text(draw, (70, top + plot_h // 2), "Mean F1", 27, DARK, True, "mm")
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
        "Text Reranker": (35, -28),
        "Fusion": (25, 42),
        "No Reranker": (25, -22),
        "Multimodal Reranker": (-220, 45),
    }

    for label, latency, f1, color in points:
        x, y = sx(latency), sy(f1)
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=color)
        dx, dy = label_offsets[label]
        draw_text(draw, (x + dx, y + dy), label, 25, DARK, True)
        body.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{color}"/>')
        body.append(svg_text(x + dx, y + dy + 22, label, 25, DARK, "700"))

    img.save(png, dpi=(300, 300))
    save_svg(svg, w, h, body)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()

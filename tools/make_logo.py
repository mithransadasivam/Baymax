"""Draw Baymax's logo: a soft white face (two dots joined by a line) on a warm red tile.

Writes baymax/assets/baymax.svg, baymax.png, and a multi-size baymax.ico. The SVG is the source
of truth for the proportions; the PNG/ICO are drawn at 4x and scaled down so edges stay smooth.

    uv run --with pillow python tools/make_logo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).resolve().parent.parent / "baymax" / "assets"
N = 1024  # design grid

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" role="img" aria-label="Baymax">
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#f05a55"/><stop offset="1" stop-color="#c32233"/>
    </linearGradient>
    <radialGradient id="face" cx="0.42" cy="0.36" r="0.75">
      <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#ece9e6"/>
    </radialGradient>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="16"/></filter>
  </defs>
  <rect width="1024" height="1024" rx="230" fill="url(#tile)"/>
  <circle cx="512" cy="540" r="330" fill="#7a0f1a" opacity="0.35" filter="url(#soft)"/>
  <circle cx="512" cy="512" r="330" fill="url(#face)"/>
  <line x1="337" y1="505" x2="687" y2="505" stroke="#1d1b1b" stroke-width="15" stroke-linecap="round"/>
  <ellipse cx="337" cy="505" rx="36" ry="42" fill="#1d1b1b"/>
  <ellipse cx="687" cy="505" rx="36" ry="42" fill="#1d1b1b"/>
</svg>
"""


def draw(size: int = N) -> Image.Image:
    s = 4  # supersample
    big = N * s
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))

    # Tile with a vertical gradient, clipped to a rounded square.
    gradient = Image.new("RGBA", (big, big))
    top, bottom = (240, 90, 85), (195, 34, 51)
    px = ImageDraw.Draw(gradient)
    for y in range(big):
        t = y / (big - 1)
        px.line([(0, y), (big, y)], fill=tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,))
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, big - 1, big - 1], radius=230 * s, fill=255)
    img.paste(gradient, (0, 0), mask)

    # Soft shadow under the face.
    shadow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([(512 - 330) * s, (540 - 330) * s, (512 + 330) * s, (540 + 330) * s], fill=(122, 15, 26, 90))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(16 * s)))

    # Face with a gentle top-left highlight.
    face = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(face)
    r = 330 * s
    for i in range(60):  # concentric discs from edge colour to highlight
        t = i / 59
        colour = tuple(round(236 + (255 - 236) * t) if c < 2 else round(233 + (255 - 233) * t) for c in range(3))
        rr = r * (1 - 0.55 * t)
        cx, cy = 512 * s - 60 * s * t, 512 * s - 70 * s * t
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=colour + (255,))
    face_mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(face_mask).ellipse([(512 - 330) * s, (512 - 330) * s, (512 + 330) * s, (512 + 330) * s], fill=255)
    img.paste(face, (0, 0), face_mask)

    # The face: two eyes joined by a line.
    d = ImageDraw.Draw(img)
    ink = (29, 27, 27, 255)
    d.line([(337 * s, 505 * s), (687 * s, 505 * s)], fill=ink, width=15 * s)
    for cx in (337, 687):
        d.ellipse([(cx - 36) * s, (505 - 42) * s, (cx + 36) * s, (505 + 42) * s], fill=ink)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "baymax.svg").write_text(SVG, encoding="utf-8")
    image = draw()
    image.resize((512, 512), Image.LANCZOS).save(ASSETS / "baymax.png")
    image.save(ASSETS / "baymax.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)])
    print("wrote", [p.name for p in sorted(ASSETS.glob("baymax.*"))])


if __name__ == "__main__":
    main()

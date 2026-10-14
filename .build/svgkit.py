"""Tiny SVG drawing helper for the lab figures (consistent house style).

Figures sit on their own light card (explicit background) so they stay readable
in both light and dark notebook themes.
"""
from html import escape
from pathlib import Path

SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, LINE, CARD = "#0f172a", "#475569", "#94a3b8", "#ffffff"
KINDS = {  # stroke, fill
    "blue": ("#2563eb", "#dbeafe"), "green": ("#16a34a", "#dcfce7"),
    "orange": ("#ea580c", "#ffedd5"), "purple": ("#7c3aed", "#ede9fe"),
    "gray": ("#64748b", "#f1f5f9"), "red": ("#dc2626", "#fee2e2"),
    "yellow": ("#ca8a04", "#fef9c3"), "teal": ("#0d9488", "#ccfbf1"),
    "white": ("#94a3b8", "#ffffff"), "pad": ("#cbd5e1", "#f8fafc"),
}


def text_w(s, size=13, mono=False):
    return len(s) * size * (0.61 if mono else 0.56)


class SVG:
    def __init__(self, w, h, title=""):
        self.w, self.h, self.title, self.items = w, h, title, []

    # ------------------------------------------------------------ primitives
    def text(self, x, y, s, size=13, anchor="start", color=INK, weight="normal",
             mono=False, italic=False):
        lines = str(s).split("\n")
        fam = MONO if mono else SANS
        style = ' font-style="italic"' if italic else ""
        dy0 = -(len(lines) - 1) * size * 0.6
        spans = "".join(
            f'<tspan x="{x}" dy="{dy0 if i == 0 else size * 1.2:.1f}">{escape(l)}</tspan>'
            for i, l in enumerate(lines))
        self.items.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="middle" xml:space="preserve"{style}>{spans}</text>')

    def rect(self, x, y, w, h, kind="blue", rx=8, dashed=False, sw=1.5):
        stroke, fill = KINDS[kind]
        d = ' stroke-dasharray="5 4"' if dashed else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def box(self, x, y, w, h, label="", kind="blue", size=13, mono=False, bold=False,
            sub=None, rx=8, dashed=False):
        self.rect(x, y, w, h, kind, rx, dashed)
        cy = y + h / 2 - (8 if sub else 0)
        if label:
            self.text(x + w / 2, cy, label, size, "middle", INK, "bold" if bold else "normal", mono)
        if sub:
            self.text(x + w / 2, y + h - 14, sub, size - 2, "middle", MUTED, mono=mono)

    def line(self, x1, y1, x2, y2, color=LINE, dashed=False, sw=1.5):
        d = ' stroke-dasharray="5 4"' if dashed else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, label=None, color="#64748b", dashed=False, lx=None, ly=None,
              size=11, path=None):
        d = ' stroke-dasharray="5 4"' if dashed else ""
        geom = path or f"M{x1},{y1} L{x2},{y2}"
        self.items.append(f'<path d="{geom}" fill="none" stroke="{color}" stroke-width="1.6"'
                          f'{d} marker-end="url(#ah)"/>')
        if label:
            lx = lx if lx is not None else (x1 + x2) / 2
            ly = ly if ly is not None else (y1 + y2) / 2 - 10
            self.text(lx, ly, label, size, "middle", MUTED, italic=True)

    def chip(self, x, y, s, kind="blue", h=26, size=12, mono=True, w=None, bold=False):
        w = w or max(26, text_w(s, size, mono) + 14)
        self.rect(x, y, w, h, kind, rx=5, sw=1.2)
        self.text(x + w / 2, y + h / 2 + 1, s, size, "middle", INK, "bold" if bold else "normal", mono)
        return w

    def chips(self, x, y, tokens, kind="blue", gap=5, h=26, size=12, kinds=None, w=None):
        """Draw a row of token chips; returns the x after the last chip."""
        for i, t in enumerate(tokens):
            k = kinds[i] if kinds else kind
            x += self.chip(x, y, t, k, h, size, w=w) + gap
        return x - gap

    def grid(self, x, y, rows, cw=34, ch=26, kind_of=None, size=11):
        """rows = list of lists of strings; kind_of(r, c, value) -> kind."""
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                k = kind_of(r, c, v) if kind_of else "white"
                self.rect(x + c * cw, y + r * ch, cw - 3, ch - 3, k, rx=4, sw=1)
                self.text(x + c * cw + (cw - 3) / 2, y + r * ch + (ch - 3) / 2 + 1, v, size,
                          "middle", INK, mono=True)

    def title_bar(self, s, sub=None):
        self.text(20, 26, s, 16, weight="bold")
        if sub:
            self.text(20, 48, sub, 12, color=MUTED)

    def note(self, x, y, s, size=11, anchor="start", color=MUTED):
        self.text(x, y, s, size, anchor, color, italic="\\" not in s)   # backslashes look like | in italics

    # ------------------------------------------------------------ output
    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img" aria-label="{escape(self.title)}">'
                f'<title>{escape(self.title)}</title>'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
                'fill="#64748b"/></marker></defs>'
                f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="12" '
                f'fill="{CARD}" stroke="#e2e8f0"/>')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(head + "".join(self.items) + "</svg>\n", encoding="utf-8")

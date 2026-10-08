# ruff: noqa: E501  (long SVG strings and diagram coordinates read better on one line)
"""Generate the end-to-end architecture diagram in docs/ (light and dark variants).

Standard library only. Run from the repository root:

    python docs/diagrams/generate_diagrams.py

Writes docs/architecture.svg and docs/architecture-dark.svg, used by the README.
Arrowheads are drawn as explicit triangles because some SVG viewers drop <marker>.
"""

import math
from html import escape
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent

THEMES = {
    "light": {
        "bg": "#ffffff",
        "box": "#f3f5fb",
        "box2": "#fafbff",
        "group": "#fafbff",
        "group_stroke": "#9aa3d9",
        "stroke": "#3f4ab8",
        "title": "#14213d",
        "text": "#4b587c",
        "accent": "#3f4ab8",
        "green": "#2e7d4f",
        "green_fill": "#eef7f1",
        "amber": "#9a6700",
        "amber_fill": "#fff8e6",
        "red": "#b42318",
        "red_fill": "#fdf0ef",
        "data_fill": "#eef6fb",
        "data": "#0b6f8a",
        "ai": "#7a3fb8",
        "ai_fill": "#f7f2fd",
        "muted": "#7a849e",
        "aws_fill": "#f6f7f9",
        "aws_stroke": "#b4bac7",
    },
    "dark": {
        "bg": "#0d1117",
        "box": "#161b22",
        "box2": "#11161d",
        "group": "#11161d",
        "group_stroke": "#4a5280",
        "stroke": "#8b93e8",
        "title": "#e6edf3",
        "text": "#9aa4b2",
        "accent": "#8b93e8",
        "green": "#56c88a",
        "green_fill": "#132a1d",
        "amber": "#e3b341",
        "amber_fill": "#2b2410",
        "red": "#f47067",
        "red_fill": "#2d1414",
        "data_fill": "#0f2430",
        "data": "#4fb3cf",
        "ai": "#c297f0",
        "ai_fill": "#1e1530",
        "muted": "#7d8796",
        "aws_fill": "#141920",
        "aws_stroke": "#3b4350",
    },
}


class Svg:
    def __init__(self, w, h, t, label):
        self.w, self.h, self.t, self.label, self.parts = w, h, t, label, []
        self.back = []  # group frames, drawn beneath boxes and arrows

    def rect(self, x, y, w, h, fill, stroke, dash=False, rx=12, sw=2):
        d = ' stroke-dasharray="7 6"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def text(self, x, y, s, size=15, weight=400, color=None, anchor="middle"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color or self.t["text"]}" text-anchor="{anchor}">{escape(s)}</text>'
        )

    def box(
        self,
        x,
        y,
        w,
        h,
        title,
        lines=(),
        title_color=None,
        fill=None,
        stroke=None,
        dash=False,
        tsize=18,
        lsize=14.5,
        sw=2,
    ):
        self.rect(x, y, w, h, fill or self.t["box"], stroke or self.t["stroke"], dash, sw=sw)
        n = 1 + len(lines)
        top = y + h / 2 - (n - 1) * 11 + 6
        self.text(x + w / 2, top, title, tsize, 700, title_color or self.t["title"])
        for i, ln in enumerate(lines):
            self.text(x + w / 2, top + 24 + i * 21, ln, lsize)

    def aws_box(self, x, y, w, h, title, lines=(), dash=False):
        """Infrastructure context, drawn quieter than the request path."""
        self.box(
            x,
            y,
            w,
            h,
            title,
            lines,
            fill=self.t["aws_fill"],
            stroke=self.t["aws_stroke"],
            dash=dash,
            tsize=15.5,
            lsize=13.5,
            sw=1.5,
        )

    def group(self, x, y, w, h, label, color, fill=None):
        parts, self.parts = self.parts, self.back
        self.rect(x, y, w, h, fill or self.t["group"], color, dash=True, rx=18)
        self.text(x + 22, y + 28, label, 15, 700, color, "start")
        self.parts = parts

    def zone(self, x, y, w, h, label, color=None):
        c = color or self.t["muted"]
        parts, self.parts = self.parts, self.back
        self.rect(x, y, w, h, "none", c, dash=True, rx=16, sw=1.6)
        self.text(x + 16, y + 24, label.upper(), 12.5, 700, c, "start")
        self.parts = parts

    def arrow(self, pts, color=None, label=None, lx=None, ly=None, dash=False, anchor="middle"):
        c = color or self.t["accent"]
        d = ' stroke-dasharray="7 6"' if dash else ""
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1)
        hl, hw = 13, 7
        bx, by = x2 - hl * math.cos(a), y2 - hl * math.sin(a)
        line = [*list(pts[:-1]), (bx, by)]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(line))
        self.parts.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2.5"{d}/>')
        p1 = (bx + hw * math.sin(a), by - hw * math.cos(a))
        p2 = (bx - hw * math.sin(a), by + hw * math.cos(a))
        self.parts.append(
            f'<path d="M{x2:.1f},{y2:.1f} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} z" fill="{c}"/>'
        )
        if label:
            self.text(lx, ly, label, 13.5, 600, c, anchor)

    def render(self):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" role="img" aria-label="{escape(self.label)}">\n'
            '<style>text{font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}</style>\n'
            f'<rect width="{self.w}" height="{self.h}" fill="{self.t["bg"]}"/>\n'
            + "\n".join(self.back + self.parts)
            + "\n</svg>\n"
        )


LABEL = (
    "Policy RAG Platform end-to-end architecture. A REST API client reaches an Application Load Balancer, whose "
    "public ingress requires HTTPS with an ACM certificate, and which forwards to the FastAPI service on ECS "
    "Fargate; an MCP consumer runs the MCP server locally over stdio with its own token. Both present a bearer "
    "token that is validated against an external identity provider's JWKS keys and turned into an access scope "
    "of role, tenant, department and access level. The scope is applied inside every retrieval query, before "
    "retrieval: semantic (pgvector HNSW), lexical (PostgreSQL full-text) or hybrid (Reciprocal Rank Fusion) "
    "search against RDS PostgreSQL 16 with pgvector in private subnets, then optional cross-encoder reranking "
    "and an evidence gate. A LangGraph answer generator (extractive by default, AWS Bedrock optional) and "
    "citation validation return an answer with sources or a refusal. A runtime and operations rail shows ECR, "
    "Secrets Manager, CloudWatch Logs, optional Bedrock, least-privilege IAM, and OpenTelemetry with "
    "Prometheus; held-out evaluation and CI run across the platform."
)


def architecture(t):
    W, H = 1440, 1466
    s = Svg(W, H, t, LABEL)
    cx = 520

    # Clients
    s.box(100, 24, 300, 84, "REST API client", ["search · ask · documents"], fill=t["box2"])
    s.box(720, 24, 260, 84, "MCP consumer", ["stdio · read-only tools"], fill=t["box2"])
    s.arrow([(330, 108), (330, 182)], label="HTTPS", lx=342, ly=129, anchor="start")

    # Entry and runtime
    s.zone(70, 140, 700, 158, "Entry & runtime · AWS model")
    s.aws_box(100, 182, 300, 92, "Application Load Balancer", ["public ingress: HTTPS + ACM"])
    s.box(470, 182, 270, 92, "FastAPI", ["on ECS Fargate"], title_color=t["accent"], fill=t["box2"])
    s.arrow([(400, 228), (470, 228)])

    # Identity and authorization
    s.arrow([(605, 274), (605, 340)], label="bearer token", lx=617, ly=314, anchor="start")
    s.arrow([(850, 108), (850, 340)], label="own token", lx=862, ly=226, anchor="start")
    s.box(
        160,
        340,
        760,
        108,
        "JWT validation → access scope",
        ["OIDC/JWKS or PEM key · algorithm allowlist", "role · tenant · department · access level"],
        title_color=t["amber"],
        fill=t["amber_fill"],
        stroke=t["amber"],
    )
    s.aws_box(
        1040, 348, 360, 92, "External identity provider", ["OIDC · JWKS signing keys"], dash=True
    )
    s.arrow([(1040, 394), (920, 394)], color=t["muted"], label="JWKS", lx=980, ly=384)

    s.arrow(
        [(cx, 448), (cx, 528)],
        color=t["amber"],
        label="authorization applied before retrieval",
        lx=cx + 14,
        ly=508,
        anchor="start",
    )

    # Authorized scope
    s.group(60, 466, 940, 894, "Authorized scope: permitted chunks only", t["amber"])

    # Retrieval
    s.rect(100, 528, 860, 176, t["box2"], t["stroke"], rx=14)
    s.text(122, 556, "Retrieval", 16, 700, t["accent"], "start")
    for i, (title, lines) in enumerate(
        [
            ("Semantic", ["pgvector HNSW", "cosine similarity"]),
            ("Lexical", ["PostgreSQL full-text", "tsvector + GIN"]),
            ("Hybrid", ["semantic + lexical", "Reciprocal Rank Fusion"]),
        ]
    ):
        s.box(122 + i * 279, 572, 259, 112, title, lines)

    # Data
    s.group(1030, 528, 380, 176, "Data · private subnets", t["data"], fill=t["data_fill"])
    s.box(
        1050,
        568,
        340,
        116,
        "RDS PostgreSQL 16 + pgvector",
        ["documents · chunks · embeddings", "authorization metadata"],
        title_color=t["data"],
        fill=t["bg"],
        stroke=t["data"],
        tsize=16.5,
    )
    s.arrow([(960, 626), (1050, 626)], color=t["data"], label="SQL", lx=1015, ly=614)

    # Rerank and gate
    s.arrow([(cx, 704), (cx, 740)])
    s.box(300, 740, 440, 76, "Cross-encoder reranking", ["optional · bounded candidate set"])
    s.arrow([(cx, 816), (cx, 852)])
    s.box(
        300,
        852,
        440,
        80,
        "Evidence gate",
        ["best similarity ≥ threshold?"],
        title_color=t["accent"],
        fill=t["box2"],
    )

    # Grounded generation
    s.arrow(
        [(cx, 932), (cx, 1012)], label="sufficient evidence", lx=cx + 14, ly=962, anchor="start"
    )
    s.group(
        100, 978, 860, 232, "Grounded generation: gated chunks only", t["ai"], fill=t["ai_fill"]
    )
    s.box(
        200,
        1012,
        640,
        80,
        "LangGraph answer generator",
        ["extractive (default) · Bedrock or OpenAI-compatible (optional)"],
        title_color=t["ai"],
        fill=t["bg"],
        stroke=t["ai"],
    )
    s.arrow([(cx, 1092), (cx, 1116)], color=t["ai"])
    s.box(
        300,
        1116,
        440,
        72,
        "Citation validation",
        ["every [n] must match a supplied chunk"],
        title_color=t["ai"],
        fill=t["bg"],
        stroke=t["ai"],
    )

    # Outcomes
    s.box(
        130,
        1250,
        380,
        80,
        "Refusal",
        ["no sources · reason returned"],
        title_color=t["red"],
        fill=t["red_fill"],
        stroke=t["red"],
    )
    s.box(
        580,
        1250,
        360,
        80,
        "Answer + sources",
        ["cited chunks · document references"],
        title_color=t["green"],
        fill=t["green_fill"],
        stroke=t["green"],
    )
    s.arrow(
        [(420, 1188), (420, 1250)],
        color=t["red"],
        label="no valid citation",
        lx=408,
        ly=1230,
        anchor="end",
    )
    s.arrow(
        [(640, 1188), (640, 1250)],
        color=t["green"],
        label="valid citations",
        lx=652,
        ly=1230,
        anchor="start",
    )
    s.arrow(
        [(300, 892), (80, 892), (80, 1290), (130, 1290)],
        color=t["red"],
        label="insufficient evidence",
        lx=190,
        ly=882,
    )

    # AWS runtime and operations rail
    s.zone(1030, 728, 380, 632, "AWS runtime & operations")
    rail = [
        ("Amazon ECR", ["application image"], False),
        ("Secrets Manager", ["database credentials"], False),
        ("CloudWatch Logs", ["application logs"], False),
        ("Amazon Bedrock", ["optional generation · listed models"], True),
        ("IAM", ["least-privilege task roles"], False),
        ("OpenTelemetry · Prometheus", ["traces · metrics · JSON logs"], False),
    ]
    for i, (title, lines, dash) in enumerate(rail):
        s.aws_box(1050, 768 + i * 96, 340, 76, title, lines, dash=dash)
    s.arrow([(840, 1075), (1050, 1075)], color=t["ai"], dash=True)
    s.text(900, 1065, "optional", 12.5, 600, t["ai"])

    # Across the platform
    s.group(60, 1378, 1350, 76, "Across the platform", t["group_stroke"])
    for i, (title, line) in enumerate(
        [
            ("Held-out evaluation", "Hit@K · MRR · nDCG floors in CI"),
            ("GitHub Actions CI", "tests · types · Terraform · security scans"),
        ]
    ):
        s.box(330 + i * 540, 1388, 500, 56, title, [line], tsize=16, lsize=13.5)
    return s.render()


def main():
    for theme, palette in THEMES.items():
        suffix = "" if theme == "light" else "-dark"
        (DOCS / f"architecture{suffix}.svg").write_text(architecture(palette), encoding="utf8")
    print("diagrams written to", DOCS)


if __name__ == "__main__":
    main()

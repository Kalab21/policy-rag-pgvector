# ruff: noqa: E501  (long SVG strings and diagram coordinates read better on one line)
"""Generate the architecture diagrams in docs/ (light and dark variants).

Standard library only. Run from the repository root:

    python docs/diagrams/generate_diagrams.py

Writes docs/architecture.svg, docs/architecture-dark.svg,
docs/aws-reference-deployment.svg and docs/aws-reference-deployment-dark.svg.
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
        self, x, y, w, h, title, lines=(), title_color=None, fill=None, stroke=None, dash=False
    ):
        self.rect(x, y, w, h, fill or self.t["box"], stroke or self.t["stroke"], dash)
        n = 1 + len(lines)
        top = y + h / 2 - (n - 1) * 11 + 6
        self.text(x + w / 2, top, title, 18, 700, title_color or self.t["title"])
        for i, ln in enumerate(lines):
            self.text(x + w / 2, top + 24 + i * 21, ln, 14.5)

    def group(self, x, y, w, h, label, color, fill=None):
        parts, self.parts = self.parts, self.back
        self.rect(x, y, w, h, fill or self.t["group"], color, dash=True, rx=18)
        self.text(x + 22, y + 28, label, 15, 700, color, "start")
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


def architecture(t):
    W, H = 1400, 1500
    s = Svg(
        W,
        H,
        t,
        (
            "Policy RAG Platform logical architecture. A REST API client or MCP consumer sends a bearer token; "
            "JWT validation produces an access scope (roles, tenant, department, access level) that is applied inside "
            "every retrieval query against PostgreSQL 16 with pgvector. Semantic, lexical or hybrid retrieval, optional "
            "cross-encoder reranking and an evidence gate run only on authorized chunks; LangGraph generation and "
            "citation validation return an answer with sources or a refusal. OpenTelemetry, Prometheus, held-out "
            "evaluation and CI run across the platform."
        ),
    )
    cx = 525  # centre of the request path

    # 1. Client
    s.box(
        225,
        30,
        600,
        100,
        "Client  /  MCP consumer",
        ["REST API: search · ask · documents", "MCP tool server (stdio): read-only, bounded tools"],
        fill=t["box2"],
    )
    s.arrow([(cx, 130), (cx, 184)], label="bearer token (JWT)", lx=cx + 14, ly=163, anchor="start")

    # 2. Authentication and access scope
    s.box(
        175,
        184,
        700,
        112,
        "Authentication + access scope",
        [
            "JWT validation: OIDC/JWKS or PEM key · algorithm allowlist",
            "→ roles · tenant · department · highest access level",
        ],
        title_color=t["amber"],
        fill=t["amber_fill"],
        stroke=t["amber"],
    )
    s.arrow(
        [(cx, 296), (cx, 400)],
        color=t["amber"],
        label="access scope, applied inside every query",
        lx=cx + 14,
        ly=356,
        anchor="start",
    )

    # Trust boundary: everything after authentication sees only authorized chunks
    s.group(60, 330, 930, 850, "Authorized scope: permitted chunks only", t["amber"])

    # 3. Retrieval layer
    s.rect(100, 400, 850, 196, t["box2"], t["stroke"], rx=14)
    s.text(122, 428, "Retrieval layer", 16, 700, t["accent"], "start")
    for i, (title, lines) in enumerate(
        [
            ("Semantic (vector)", ["pgvector HNSW", "cosine similarity"]),
            ("Lexical", ["PostgreSQL full-text", "tsvector + GIN"]),
            ("Hybrid", ["semantic + lexical", "Reciprocal Rank Fusion"]),
        ]
    ):
        s.box(122 + i * 276, 446, 254, 128, title, lines)

    # Data layer, beside the retrieval layer
    s.group(1030, 400, 310, 196, "Data", t["data"], fill=t["data_fill"])
    s.box(
        1052,
        446,
        266,
        128,
        "PostgreSQL 16",
        ["+ pgvector", "chunks · embeddings", "tenant · dept · level"],
        title_color=t["data"],
        fill=t["bg"],
        stroke=t["data"],
    )
    s.arrow([(950, 510), (1052, 510)], color=t["data"], label="SQL", lx=990, ly=500)

    # 4. Reranker
    s.arrow([(cx, 596), (cx, 640)])
    s.box(
        305,
        640,
        440,
        86,
        "Cross-encoder reranking (optional)",
        ["re-orders a bounded candidate set"],
    )

    # 5. Evidence gate
    s.arrow([(cx, 726), (cx, 770)])
    s.box(
        305,
        770,
        440,
        92,
        "Evidence gate",
        ["best similarity ≥ threshold?"],
        title_color=t["accent"],
        fill=t["box2"],
    )

    # 6/7. AI generation boundary
    s.arrow([(cx, 862), (cx, 940)], label="sufficient evidence", lx=cx + 14, ly=890, anchor="start")
    s.group(
        100, 908, 850, 248, "AI generation boundary: gated chunks only", t["ai"], fill=t["ai_fill"]
    )
    s.box(
        220,
        950,
        610,
        92,
        "LangGraph generation",
        ["extractive (default) · Bedrock / OpenAI-compatible (optional)"],
        title_color=t["ai"],
        fill=t["bg"],
        stroke=t["ai"],
    )
    s.arrow([(cx, 1042), (cx, 1066)], color=t["ai"])
    s.box(
        305,
        1066,
        440,
        72,
        "Citation validation",
        ["every [n] must match a supplied chunk"],
        title_color=t["ai"],
        fill=t["bg"],
        stroke=t["ai"],
    )

    # Outcomes
    s.arrow(
        [(cx, 1138), (cx, 1226)],
        color=t["green"],
        label="valid citations",
        lx=cx + 14,
        ly=1206,
        anchor="start",
    )
    s.box(
        305,
        1226,
        440,
        84,
        "Answer with sources",
        ["cited chunks and document references"],
        title_color=t["green"],
        fill=t["green_fill"],
        stroke=t["green"],
    )
    s.box(
        1040,
        1226,
        300,
        84,
        "Refusal",
        ["no sources · reason returned"],
        title_color=t["red"],
        fill=t["red_fill"],
        stroke=t["red"],
    )
    s.arrow(
        [(745, 816), (1250, 816), (1250, 1226)],
        color=t["red"],
        label="insufficient evidence",
        lx=1000,
        ly=806,
    )
    s.arrow(
        [(745, 1102), (1120, 1102), (1120, 1226)],
        color=t["red"],
        label="no valid citation",
        lx=1052,
        ly=1092,
    )

    # Cross-cutting operations strip
    s.group(60, 1350, 1280, 130, "Across the platform", t["group_stroke"])
    for i, (title, line) in enumerate(
        [
            ("OpenTelemetry", "traces per pipeline stage"),
            ("Prometheus + JSON logs", "no question or document text"),
            ("Evaluation", "held-out + tuning sets"),
            ("GitHub Actions CI", "tests · types · IaC · security"),
        ]
    ):
        s.box(84 + i * 312, 1390, 296, 72, title, [line])
    return s.render()


def aws(t):
    W, H = 1400, 960
    s = Svg(
        W,
        H,
        t,
        (
            "AWS reference deployment defined in Terraform and not currently deployed. Clients on allowed networks "
            "reach an Application Load Balancer over HTTPS with an ACM certificate (required for a public load "
            "balancer). The load balancer forwards to an ECS Fargate service, which connects to RDS PostgreSQL 16 "
            "with pgvector in private subnets. The task pulls its image from ECR, reads database credentials from "
            "Secrets Manager, writes logs to CloudWatch, runs under least-privilege IAM roles, verifies tokens against "
            "the identity provider's JWKS endpoint and may call Bedrock only if model ARNs are granted."
        ),
    )
    s.box(
        400,
        30,
        600,
        92,
        "Clients on allowed networks",
        ["alb_ingress_cidrs (no default)"],
        fill=t["box2"],
    )
    s.arrow(
        [(700, 122), (700, 214)],
        label="HTTPS (ACM certificate, TLS 1.3 policy)",
        lx=714,
        ly=152,
        anchor="start",
    )

    s.group(
        60, 170, 860, 760, "VPC (Terraform reference, 2-3 availability zones)", t["group_stroke"]
    )
    s.box(
        420,
        214,
        460,
        104,
        "Application Load Balancer",
        ["internal by default · public needs a certificate", "HTTP redirects to HTTPS"],
        title_color=t["accent"],
        fill=t["box2"],
    )
    s.arrow(
        [(650, 318), (650, 420)],
        label="container port, from the ALB only",
        lx=664,
        ly=376,
        anchor="start",
    )

    s.group(
        100, 384, 780, 220, "Application subnets (private with the NAT gateway option)", t["accent"]
    )
    s.box(
        300,
        430,
        480,
        150,
        "ECS Fargate service",
        [
            "FastAPI task · AUTH_MODE=jwt",
            "read-only root filesystem",
            "egress: HTTPS + PostgreSQL only",
        ],
    )
    s.arrow(
        [(540, 580), (540, 690)],
        color=t["data"],
        label="PostgreSQL, from the app only",
        lx=554,
        ly=642,
        anchor="start",
    )

    s.group(100, 654, 780, 250, "Private data subnets", t["data"], fill=t["data_fill"])
    s.box(
        300,
        700,
        480,
        170,
        "RDS PostgreSQL 16 + pgvector",
        [
            "encrypted storage · not publicly accessible",
            "automated backups",
            "master password managed by RDS",
            "in Secrets Manager",
        ],
        title_color=t["data"],
        fill=t["bg"],
        stroke=t["data"],
    )

    # AWS services and external endpoints reached over HTTPS
    s.text(1150, 200, "Reached over HTTPS (443)", 15, 700, t["accent"])
    side = [
        ("Amazon ECR", ["image: immutable tags, scan on push"], None, None, None),
        ("Secrets Manager", ["DB_USER / DB_PASSWORD"], None, None, None),
        ("CloudWatch Logs", ["one log group"], None, None, None),
        ("Identity provider JWKS", ["external; token signature keys"], None, None, None),
        (
            "Amazon Bedrock (optional)",
            ["InvokeModel on listed ARNs only"],
            t["ai"],
            t["ai_fill"],
            t["ai"],
        ),
        (
            "IAM roles",
            ["execution + task roles, least privilege", "(attached to the task)"],
            t["amber"],
            t["amber_fill"],
            t["amber"],
        ),
    ]
    for i, (title, lines, tc, fill, stroke) in enumerate(side):
        y = 224 + i * 116
        s.box(980, y, 340, 92, title, lines, title_color=tc, fill=fill, stroke=stroke, dash=i == 4)
    # one trunk from the task to the side services
    s.parts.append(
        f'<path d="M780,505 L940,505 M940,270 L940,{224 + 4 * 116 + 46}" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="2.5"/>'
    )
    for i in range(5):
        y = 224 + i * 116 + 46
        s.arrow([(940, y), (980, y)], color=t["ai"] if i == 4 else None, dash=i == 4)
    return s.render()


def main():
    for theme, palette in THEMES.items():
        suffix = "" if theme == "light" else "-dark"
        (DOCS / f"architecture{suffix}.svg").write_text(architecture(palette), encoding="utf8")
        (DOCS / f"aws-reference-deployment{suffix}.svg").write_text(aws(palette), encoding="utf8")
    print("diagrams written to", DOCS)


if __name__ == "__main__":
    main()

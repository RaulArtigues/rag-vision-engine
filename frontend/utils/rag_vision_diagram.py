import os

def save_svg(svg_content: str, filename: str, relative_dir: str = "../../assets"):
    """
    Save an SVG string into a specified relative directory.

    This helper function writes raw SVG markup into a `.svg` file located in a
    user-defined relative folder (default: `../../assets`). It ensures that the
    directory exists, creates it if necessary, and returns the final absolute
    path to the generated file.

    Args:
        svg_content (str):
            The SVG document as a string. Must contain valid XML/SVG markup.
        filename (str):
            Name of the output SVG file (e.g., "diagram.svg").
        relative_dir (str, optional):
            Relative directory path where the SVG will be stored.
            Defaults to `"../../assets"`.

    Returns:
        str:
            The absolute path to the saved SVG file.

    Side Effects:
        - Creates the target directory if it does not already exist.
        - Writes the SVG content to disk.

    Example:
        >>> from save_svg import save_svg
        >>> save_svg("<svg>...</svg>", "architecture.svg")
        SVG saved in: /absolute/path/to/assets/architecture.svg
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, relative_dir)
    target_path = os.path.join(target_dir, filename)

    os.makedirs(target_dir, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"SVG saved in: {os.path.abspath(target_path)}")
    return target_path

svg_content = """<svg width="1100" height="340" viewBox="0 0 1100 340" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: sans-serif;">
    <rect width="1100" height="340" fill="#0b0f19" rx="12" />
    <defs>
        <linearGradient id="gradCard" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#1f2937;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#111827;stop-opacity:1" />
        </linearGradient>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1" />
        </marker>
    </defs>
    <style>
        .flow-path { fill: none; stroke: #6366f1; stroke-width: 2; stroke-dasharray: 8; animation: dash 30s linear infinite; marker-end: url(#arrowhead); }
        @keyframes dash { to { stroke-dashoffset: -1000; } }
        .box { fill: url(#gradCard); stroke: #374151; stroke-width: 1.5; rx: 10; }
        .box-active { stroke: #818cf8; stroke-width: 2; }
        .title { fill: #e5e7eb; font-weight: bold; font-size: 15px; text-anchor: middle; }
        .desc { fill: #9ca3af; font-size: 12px; text-anchor: middle; }
        .tech-label { fill: #6b7280; font-family: monospace; font-size: 11px; text-anchor: middle; font-weight: bold; }
        .icon { font-size: 24px; text-anchor: middle; dominant-baseline: middle; }
        .premise-box { stroke: #ec4899; stroke-width: 2; fill: rgba(236, 72, 153, 0.1); }
        .goal-box { stroke: #10b981; stroke-width: 2; fill: rgba(16, 185, 129, 0.1); }
    </style>

    <rect x="20" y="120" width="140" height="100" class="box premise-box" />
    <text x="90" y="155" class="icon">🎯</text>
    <text x="90" y="180" class="title">Premise</text>
    <text x="90" y="200" class="desc">Binary Classif.</text>
    <path d="M160 170 L210 170" class="flow-path" />

    <rect x="210" y="50" width="160" height="240" class="box" />
    <text x="290" y="80" class="title">1. Support Set</text>
    <rect x="230" y="100" width="120" height="50" fill="#374151" rx="5" />
    <text x="290" y="130" class="desc" fill="white">Class A (e.g. Clean)</text>
    <rect x="230" y="160" width="120" height="50" fill="#374151" rx="5" />
    <text x="290" y="190" class="desc" fill="white">Class B (e.g. Dirty)</text>
    <text x="290" y="250" class="desc" style="font-style: italic;">Uploaded Images</text>
    <path d="M370 170 L440 170" class="flow-path" />
    <text x="405" y="160" class="tech-label">Patching</text>

    <rect x="440" y="30" width="240" height="280" class="box box-active" />
    <text x="560" y="60" class="title" style="fill:#818cf8">2. RAG Engine</text>
    <text x="560" y="80" class="tech-label">Model: CLIP</text>
    <rect x="460" y="100" width="200" height="60" fill="rgba(255,255,255,0.05)" stroke="#4b5563" rx="5" />
    <text x="480" y="135" class="icon">🖼️</text>
    <text x="570" y="125" class="title" style="font-size:13px">Target Image</text>
    <text x="570" y="145" class="desc">(Query)</text>
    <rect x="480" y="190" width="160" height="80" fill="#1e1b4b" stroke="#6366f1" rx="8" />
    <text x="560" y="220" class="title" style="font-size:13px; fill:#c7d2fe">Vector Search</text>
    <text x="560" y="240" class="desc">Cosine Similarity</text>
    <text x="560" y="260" class="tech-label">k-Retrieval Matches</text>
    <path d="M680 170 L750 170" class="flow-path" />
    <text x="715" y="160" class="tech-label">Context</text>

    <rect x="750" y="30" width="180" height="280" class="box box-active" />
    <text x="840" y="60" class="title" style="fill:#c084fc">3. Inference</text>
    <text x="840" y="80" class="tech-label">Model: Qwen2-VL</text>
    <rect x="770" y="110" width="140" height="60" fill="rgba(255,255,255,0.05)" stroke="#4b5563" rx="5" />
    <text x="790" y="145" class="icon">📝</text>
    <text x="850" y="135" class="title" style="font-size:12px">Prompts</text>
    <text x="850" y="150" class="desc">Sys + User</text>
    <circle cx="840" cy="230" r="40" fill="#312e81" stroke="#a5b4fc" stroke-width="2" />
    <text x="840" y="235" class="icon" font-size="30">🧠</text>
    <path d="M930 170 L980 170" class="flow-path" />

    <rect x="980" y="100" width="100" height="140" class="box goal-box" />
    <text x="1030" y="130" class="icon">✅</text>
    <text x="1030" y="160" class="title">Result</text>
    <text x="1030" y="185" class="desc">Flag + Reason</text>
    <line x1="990" y1="200" x2="1070" y2="200" stroke="#10b981" stroke-width="1" />
    <text x="1030" y="225" class="tech-label" fill="#10b981">Auto-Labeling</text>
</svg>
"""

save_svg(svg_content, "diagrama_rag_vision.svg")
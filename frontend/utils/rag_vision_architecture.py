import base64
import os

def generate_engineering_svg(filename="rag_vision_architecture_dynamic.svg"):
    """
    Generates a high-fidelity, engineering-grade SVG diagram representing
    the RAG-Vision backend architecture with DYNAMIC lines and MATH specs.
    """
    C_BG = "#0b0f19"
    C_GRID = "#1f2937"
    C_BOX_FILL = "#111827"
    C_BOX_STROKE = "#374151"
    C_TEXT_MAIN = "#e5e7eb"
    C_TEXT_DIM = "#9ca3af"
    C_ACCENT_CYAN = "#22d3ee"
    C_ACCENT_PURPLE = "#a78bfa"
    C_ACCENT_GREEN = "#34d399"
    C_ACCENT_ORANGE = "#fb923c"
    C_CODE = "#60a5fa"
    
    svg_content = f"""
    <svg width="1440" height="840" viewBox="0 0 1440 840" xmlns="http://www.w3.org/2000/svg" style="font-family: 'IBM Plex Sans', 'Menlo', sans-serif; background-color: {C_BG};">
        <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="{C_GRID}" stroke-width="1"/>
            </pattern>
            
            <linearGradient id="gradProcess" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#1f2937;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#111827;stop-opacity:1" />
            </linearGradient>
            
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="{C_ACCENT_CYAN}" />
            </marker>
            
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            <style>
                .dynamic-flow {{
                    stroke: {C_ACCENT_CYAN};
                    stroke-width: 2;
                    stroke-dasharray: 10;
                    animation: dash 1s linear infinite;
                    fill: none;
                }}
                
                @keyframes dash {{
                    to {{
                        stroke-dashoffset: -20;
                    }}
                }}
                
                .tech-label {{
                    font-family: 'Menlo', monospace;
                    font-size: 10px;
                    fill: {C_CODE};
                }}
            </style>
        </defs>

        <!-- Background -->
        <rect width="100%" height="100%" fill="{C_BG}" />
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        <!-- Header -->
        <g transform="translate(40, 0)">
            
            <!-- Accent bar -->
            <rect x="0" y="15" width="6" height="90" fill="{C_ACCENT_CYAN}" rx="2" />

            <!-- Title -->
            <text 
                x="20" y="40" 
                fill="{C_TEXT_MAIN}" 
                font-size="28" 
                font-weight="bold"
                letter-spacing="1"
            >
                RAG-VISION ENGINE ARCHITECTURE
            </text>

            <!-- Subtitle 1 -->
            <text 
                x="20" y="70" 
                fill="{C_ACCENT_PURPLE}" 
                font-size="14" 
                font-family="monospace"
            >
                Dynamic Pipeline Visualization v1.0 · ViT-B/32 + Qwen2-VL
            </text>

            <!-- Subtitle 2 -->
            <text 
                x="20" y="100" 
                fill="{C_TEXT_DIM}" 
                font-size="14" 
                font-family="monospace"
            >
                Binary Classification · Semi-Automatic Image Labeling Workflow
            </text>
        </g>

        <!-- ===================== OFFLINE PHASE ===================== -->
        <rect x="40" y="120" width="400" height="680" rx="10" fill="rgba(17, 24, 39, 0.5)" stroke="{C_BOX_STROKE}" stroke-dasharray="5,5"/>
        <text x="60" y="150" fill="{C_ACCENT_ORANGE}" font-weight="bold" font-size="14">OFFLINE PHASE: Vector Indexing</text>
        
        <!-- Class Dirs -->
        <g transform="translate(90, 0)">
            <rect x="80" y="180" width="140" height="80" rx="4"
                fill="{C_BOX_FILL}" stroke="{C_ACCENT_ORANGE}" stroke-width="2"/>
            
            <text x="150" y="215" fill="{C_TEXT_MAIN}" 
                text-anchor="middle" font-size="14">Raw Dataset</text>
            
            <text x="150" y="235" fill="{C_TEXT_DIM}" 
                text-anchor="middle" font-size="10">Class A / Class B</text>
            
            <text x="150" y="200" fill="{C_TEXT_DIM}" 
                text-anchor="middle" font-size="20">📂</text>
        </g>

        <!-- Patch Extraction -->
        <rect x="80" y="300" width="320" height="120" rx="4" fill="{C_BOX_FILL}" stroke="{C_BOX_STROKE}"/>
        <text x="100" y="325" fill="{C_TEXT_MAIN}" font-size="12" font-weight="bold">Patch Grid Computation</text>
        
        <!-- Grid Visual -->
        <g transform="translate(100, 340)">
            <rect width="60" height="60" fill="none" stroke="{C_ACCENT_CYAN}" stroke-width="1"/>
            <line x1="20" y1="0" x2="20" y2="60" stroke="{C_ACCENT_CYAN}" opacity="0.5"/>
            <line x1="40" y1="0" x2="40" y2="60" stroke="{C_ACCENT_CYAN}" opacity="0.5"/>
            <line x1="0" y1="20" x2="60" y2="20" stroke="{C_ACCENT_CYAN}" opacity="0.5"/>
            <line x1="0" y1="40" x2="60" y2="40" stroke="{C_ACCENT_CYAN}" opacity="0.5"/>
            <text x="30" y="75" fill="{C_TEXT_DIM}" font-size="10" text-anchor="middle">S x S Grid</text>
        </g>
        <text x="180" y="375" fill="{C_TEXT_DIM}" font-size="20">→</text>
        <text x="210" y="375" fill="{C_TEXT_MAIN}" font-size="12">Flattened Patches</text>

        <!-- CLIP Encoder -->
        <rect x="150" y="460" width="180" height="80" rx="20" fill="#1e1b4b" stroke="{C_ACCENT_CYAN}" stroke-width="2"/>
        <text x="240" y="485" fill="{C_TEXT_MAIN}" text-anchor="middle" font-weight="bold">Visual Encoder</text>
        <text x="240" y="505" class="tech-label" text-anchor="middle">CLIP ViT-B/32</text>
        <text x="240" y="525" class="tech-label" text-anchor="middle">Output: R^512</text>

        <!-- Vector Store -->
        <path d="M 120 580 L 360 580 L 360 670 L 120 670 Z" fill="{C_BOX_FILL}" stroke="{C_ACCENT_CYAN}" stroke-width="2"/>
        <line x1="120" y1="610" x2="360" y2="610" stroke="{C_BOX_STROKE}"/>
        <line x1="120" y1="640" x2="360" y2="640" stroke="{C_BOX_STROKE}"/>
        <text x="240" y="600" fill="{C_TEXT_MAIN}" text-anchor="middle" font-size="12" font-weight="bold">Tensor Index (RAM)</text>
        <text x="240" y="630" class="tech-label" text-anchor="middle">Matrix M [N, 512]</text>
        <text x="240" y="660" fill="{C_TEXT_DIM}" text-anchor="middle" font-size="10">Norm(L2) Applied</text>

        <!-- DYNAMIC LINES (Offline) -->
        <path d="M 240 260 L 240 300" class="dynamic-flow" marker-end="url(#arrow)"/>
        <path d="M 240 420 L 240 460" class="dynamic-flow" marker-end="url(#arrow)"/>
        <path d="M 240 540 L 240 580" class="dynamic-flow" marker-end="url(#arrow)"/>

        <!-- ===================== ONLINE PHASE ===================== -->
        <rect x="480" y="120" width="920" height="680" rx="10" fill="rgba(17, 24, 39, 0.5)" stroke="{C_BOX_STROKE}" stroke-dasharray="5,5"/>
        <text x="500" y="150" fill="{C_ACCENT_PURPLE}" font-weight="bold" font-size="14">ONLINE PHASE: Inference Engine</text>

        <!-- Input Blocks -->
        <g transform="translate(520, 180)">
            <rect width="120" height="80" rx="4" fill="{C_BOX_FILL}" stroke="{C_TEXT_MAIN}"/>
            <text x="60" y="30" fill="{C_TEXT_MAIN}" text-anchor="middle" font-size="12">Target Image</text>
            <text x="60" y="50" fill="{C_TEXT_DIM}" text-anchor="middle" font-size="10">Base64 Stream</text>
        </g>

        <!-- Input Blocks -->
        <g transform="translate(1020, 180)">            

        <rect width="120" height="80" rx="4"
            fill="{C_BOX_FILL}" stroke="{C_TEXT_MAIN}"/>

        <text x="60" y="30"
            fill="{C_TEXT_MAIN}"
            text-anchor="middle"
            font-size="12">
            Prompt Stream
        </text>

        <text x="60" y="50"
            fill="{C_TEXT_DIM}"
            text-anchor="middle"
            font-size="10">
            System Prompt
        </text>

        <text x="60" y="70"
            fill="{C_TEXT_DIM}"
            text-anchor="middle"
            font-size="10">
            User Prompt
        </text>

        </g>

        <!-- Preprocessor -->
        <rect x="680" y="180" width="160" height="80" rx="4"
            fill="{C_BOX_FILL}" stroke="{C_CODE}"/>

        <!-- Title -->
        <text x="760" y="210"
            fill="{C_TEXT_MAIN}"
            text-anchor="middle"
            font-size="12">
            Normalization
        </text>

        <!-- Subtitle 1 -->
        <text x="760" y="230"
            class="tech-label"
            text-anchor="middle">
            Resize(224,224)
        </text>

        <!-- Subtitle 2 -->
        <text x="760" y="250"
            class="tech-label"
            text-anchor="middle">
            RGB Convert
        </text>

        <!-- CLIP Query Encode -->
        <rect x="690" y="310" width="140" height="60" rx="30" fill="#1e1b4b" stroke="{C_ACCENT_CYAN}" stroke-width="2"/>
        <text x="760" y="345" fill="{C_TEXT_MAIN}" text-anchor="middle" font-size="11">Query Embedding</text>

        <!-- Retrieval Logic -->
        <g transform="translate(640, 420)">
            <rect width="240" height="120" rx="4" fill="#111827" stroke="{C_ACCENT_CYAN}" stroke-width="2"/>
            <text x="120" y="30" fill="{C_ACCENT_CYAN}" text-anchor="middle" font-weight="bold">RETRIEVAL KERNEL</text>
            
            <text x="120" y="60" class="tech-label" text-anchor="middle" font-size="14">sim = (S · q) / (|S||q|)</text>
            <text x="120" y="85" fill="{C_TEXT_DIM}" text-anchor="middle" font-size="10">Cosine Similarity</text>
            <text x="120" y="105" class="tech-label" text-anchor="middle">ArgSort(Top-K)</text>
        </g>

        <!-- Link Index to Retrieval -->
        <path d="M 360 620 L 540 620 L 540 480 L 640 480" class="dynamic-flow" stroke-dasharray="5,5" marker-end="url(#arrow)"/>
        <text x="410" y="610" fill="{C_ACCENT_CYAN}" font-size="10">Reference Vectors</text>

        <!-- Patch Cropping -->
        <rect x="680" y="590" width="160" height="60" rx="4"
            fill="{C_BOX_FILL}" stroke="{C_TEXT_DIM}"/>

        <!-- Title -->
        <text x="760" y="620"
            fill="{C_TEXT_MAIN}"
            text-anchor="middle"
            font-size="12">
            Evidence Extraction
        </text>

        <!-- Subtitle -->
        <text x="760" y="640"
            class="tech-label"
            text-anchor="middle">
            ROI Cropping
        </text>

        <!-- Context Window Construction -->
        <g transform="translate(960, 310)">

            <rect x="0" y="0" width="240" height="180" rx="4"
                fill="{C_BOX_FILL}" stroke="{C_ACCENT_PURPLE}"/>

            <text x="120" y="20"
                fill="{C_ACCENT_PURPLE}"
                text-anchor="middle"
                font-size="14"
                font-weight="bold">
                Multimodal Context
            </text>

            <rect x="40" y="40" width="160" height="25"
                fill="#374151" stroke="none"/>

            <text x="120" y="57"
                fill="white"
                text-anchor="middle"
                font-size="10">
                System Instructions
            </text>

            <g transform="translate(40, 75)">
                <rect width="30" height="30" fill="#1f2937" stroke="{C_ACCENT_CYAN}"/>
                <rect x="35" width="30" height="30" fill="#1f2937" stroke="{C_ACCENT_CYAN}"/>
                <rect x="70" width="30" height="30" fill="#1f2937" stroke="{C_ACCENT_CYAN}"/>

                <text x="150" y="20"
                    fill="{C_TEXT_DIM}"
                    font-size="10"
                    text-anchor="middle">
                    Visual Tokens
                </text>
            </g>

            <rect x="40" y="120" width="160" height="40"
                fill="#374151" stroke="none"/>

            <text x="120" y="145"
                fill="white"
                text-anchor="middle"
                font-size="10">
                User Query
            </text>

        </g>

        <!-- Qwen Model Node -->
        <g transform="translate(1080, 600)">
            
            <circle cx="0" cy="0" r="50"
                    fill="#312e81" stroke="{C_ACCENT_PURPLE}" stroke-width="3"/>

            <text x="0" y="0"
                  text-anchor="middle"
                  dominant-baseline="middle"
                  fill="{C_TEXT_MAIN}"
                  font-size="12">

            <tspan x="0" dy="-15" font-size="14" font-weight="bold">
                Qwen2-VL
            </tspan>

            <tspan x="0" dy="15" class="tech-label">
                2B Instruct
            </tspan>

            <tspan x="0" dy="15" class="tech-label">
                P(Y|X)
            </tspan>

        </text>

        </g>

        <!-- Output Node -->
        <g transform="translate(1280, 600)"> 

            <rect x="-80" y="-30" width="160" height="60" rx="4"
                fill="{C_BOX_FILL}" stroke="{C_ACCENT_GREEN}"/>

            <text x="0" y="-5"
                text-anchor="middle"
                dominant-baseline="middle"
                fill="{C_TEXT_MAIN}"
                font-size="12"
                font-weight="bold">
                Post-Processing
            </text>

            <text x="0" y="15"
                text-anchor="middle"
                dominant-baseline="middle"
                class="tech-label">
                Regex Parsing
            </text>

        </g>

        <!-- Final JSON -->
        <g transform="translate(1280, 725)">

            <rect x="-80" y="-45" width="160" height="80" rx="4"
                fill="rgba(52, 211, 153, 0.1)"
                stroke="{C_ACCENT_GREEN}"/>

            <text x="0" y="-15"
                fill="{C_ACCENT_GREEN}"
                text-anchor="middle"
                dominant-baseline="middle"
                font-size="14">
                JSON Result
            </text>

            <text x="0" y="5"
                fill="{C_TEXT_MAIN}"
                text-anchor="middle"
                dominant-baseline="middle"
                font-size="10">
                Flag: bool or string
            </text>

            <text x="0" y="25"
                fill="{C_TEXT_MAIN}"
                text-anchor="middle"
                dominant-baseline="middle"
                font-size="10">
                Explanation: text
            </text>

        </g>

        <!-- DYNAMIC LINES (Online Phase) -->
        <!-- Input -> Norm -->
        <path d="M 640 220 L 680 220" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Norm -> Encode -->
        <path d="M 760 260 L 760 310" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Encode -> Retrieval -->
        <path d="M 760 370 L 760 420" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Retrieval -> Evidence -->
        <path d="M 760 540 L 760 590" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Prompt -> Context -->
        <path d="M 1080 260 L 1080 310" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Evidence -> Context -->
        <path d="M 840 620 L 920 620 L 920 400 L 960 400" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Context -> Model -->
        <path d="M 1080 490 L 1080 550" class="dynamic-flow" stroke="{C_ACCENT_PURPLE}" marker-end="url(#arrow)"/>
        <!-- Model -> Post -->
        <path d="M 1130 600 L 1200 600" class="dynamic-flow" marker-end="url(#arrow)"/>
        <!-- Post -> JSON -->
        <path d="M 1280 630 L 1280 680" class="dynamic-flow" stroke="{C_ACCENT_GREEN}" marker-end="url(#arrow)"/>

        <!-- Footer (no link) -->
        <text 
            x="720" 
            y="825" 
            text-anchor="middle" 
            font-size="12" 
            fill="{C_TEXT_DIM}" 
            font-family="monospace"
        >
            Raúl Artigues | Computer Vision Engineer
        </text>

    </svg>
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    save_assets = os.path.join(script_dir, "..", "..", "assets", filename)

    os.makedirs(os.path.join(script_dir, "..","..",  "assets"), exist_ok=True)

    with open(save_assets, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print("SVG also saved to assets/ at:", os.path.abspath(save_assets))

if __name__ == "__main__":
    generate_engineering_svg()
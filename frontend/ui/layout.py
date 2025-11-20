from frontend.services.rag_api_client import upload_support_single_image, run_rag_inference
from frontend.assets.examples.example_payload import build_example_payload
from frontend.utils.text_utils import validate_english, validate_label
from frontend.utils.image_utils import read_file_as_b64
from datetime import datetime
import gradio as gr
import json

def load_svg_from_assets(filename: str) -> str:
    """
    Load an SVG file from the project's assets directory.

    This function constructs the absolute path to the SVG file based on the
    calling module's location and returns the raw SVG markup as a string.

    Args:
        filename (str):
            The name of the SVG file to be loaded (e.g., ``"diagram.svg"``).

    Returns:
        str:
            The full SVG file content as a UTF-8 string.

    Raises:
        FileNotFoundError:
            If the specified SVG file does not exist inside the assets directory.
    """
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(current_dir, "..", "..", "assets", filename)

    if not os.path.isfile(svg_path):
        raise FileNotFoundError(f"SVG not found: {svg_path}")

    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()

def build_ui():
    """
    Build and return the complete Gradio UI for the RAG-Vision Engine.

    This interface integrates:
    - A visual architectural SVG diagram loaded from assets
    - A step-by-step instructions panel
    - Inputs for defining support classes and prompts
    - Image upload components
    - Retrieval and generation configuration parameters
    - Execution controls for triggering inference
    - A postprocessing area including raw JSON results

    The UI uses custom CSS for a dark, glass-panel inspired theme and includes
    interactive elements such as accordions, galleries and responsive layouts.

    Returns:
        gr.Blocks:
            A fully constructed Gradio Blocks application that can be launched
            directly via ``ui.launch()`` or embedded inside another system.
    """
    css_style = """
        .gradio-container {
            background-color: #0b0f19 !important;
            font-family: 'IBM Plex Sans', sans-serif;
        }
        p, span, div, label, input, textarea, button, .gr-button, .prose, .prose p, .prose h3, .prose h4 {
            font-size: 16px !important; 
            line-height: 1.5;
        }

        /* HEADER */
        .depth-header {
            background: linear-gradient(135deg, #111827 0%, #1e1b4b 100%);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
            border: 1px solid #374151;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        .depth-title {
            font-size: 2.5rem !important; 
            font-weight: 800;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #818cf8, #c084fc, #e879f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }
        .depth-subtitle {
            font-size: 40px !important;
            line-height: 1.3;
            color: #d1d5db;
            margin-bottom: 1.5rem;
            font-weight: 300;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        /* CONTENEDOR DEL DIAGRAMA SVG */
        .svg-container {
            width: 100%;
            overflow-x: auto;
            margin-bottom: 2.5rem;
            display: flex;
            justify-content: center;
            background: rgba(17, 24, 39, 0.5);
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 1rem;
        }

        /* PANELES */
        .glass-panel {
            background-color: #131926 !important;
            border: 1px solid #374151;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem !important;
        }
        .params-panel {
            background-color: #1f2937 !important;
            border: 1px solid #374151;
            border-radius: 8px !important;
            padding: 1.5rem !important;
            margin-top: 1rem !important;
        }

        /* BOTONES */
        .btn-row { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
        .capsule-btn {
            display: flex; align-items: center; justify-content: center; gap: 0.5rem;
            padding: 0.6rem 1.5rem; background-color: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 9999px;
            color: #e5e7eb; text-decoration: none; font-size: 16px !important; 
            font-weight: 500; transition: all 0.2s ease;
        }
        .capsule-btn:hover {
            background-color: rgba(255, 255, 255, 0.2); border-color: rgba(255, 255, 255, 0.4);
            color: white; transform: translateY(-1px);
        }
        .favicon-icon { width: 20px; height: 20px; object-fit: contain; border-radius: 4px; }
        .capsule-btn svg { fill: currentColor; width: 20px; height: 20px; }
        
        /* ESTILOS GENERALES */
        .gr-input-label, label span, .prose h4, span.label-wrap span {
            font-size: 16px !important; font-weight: 600; color: #e5e7eb !important;
        }
        .settings-accordion .label-wrap span, .section-header h3 {
            font-size: 1.25rem !important; font-weight: 700; color: #a5b4fc !important; 
            margin-top: 0 !important; margin-bottom: 1rem !important;
        }
        .params-panel h3 { margin-top: 0.5rem !important; color: #e5e7eb !important; }
        textarea, input { font-size: 16px !important; background-color: #111827 !important; border-color: #374151 !important; }
        
        .raw-json textarea {
            font-family: "Menlo", monospace !important;
            font-size: 0.95rem !important;
            max-height: 350px !important;
            overflow-y: auto !important;
            white-space: pre-wrap !important;
        }

        input[type="number"] { min-width: 100px !important; text-align: right; }
        .gr-button { font-size: 16px !important; padding: 8px 16px; }
        
        .custom-footer {
            text-align: center; margin-top: 4rem; padding-top: 1.5rem;
            border-top: 1px solid #1f2937; color: #6b7280; font-size: 16px !important; 
        }
        .custom-footer a { color: #9ca3af; text-decoration: none; }
        .custom-footer a:hover { color: #e5e7eb; text-decoration: underline; }
        footer {visibility: hidden}
    """

    github_svg = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>"""
    
    technical_flow_svg = load_svg_from_assets("rag_vision_architecture_dynamic.svg")

    with gr.Blocks(css=css_style, theme=gr.themes.Soft()) as ui:

        gr.HTML(
            f"""
            <div class="depth-header">
                <h1 class="depth-title">RAG-Vision Engine</h1>
                <p class="depth-subtitle">
                    Retrieval-Augmented Visual Reasoning & Semi-Automatic Labeling<br>
                    <span style="opacity: 0.7;">Powered by Qwen2-VL & CLIP Retrieval</span>
                </p>
                
                <div class="btn-row">
                    <a href="https://github.com/raulartigues/rag-vision-engine" target="_blank" class="capsule-btn">
                        {github_svg}
                        View Code
                    </a>
                    <a href="https://www.raulartigues.com/en/blog" target="_blank" class="capsule-btn">
                        <img src="https://www.raulartigues.com/favicon.ico" class="favicon-icon" />
                         Case Study & Slides
                    </a>
                </div>
            </div>

            <div class="svg-container">
                {technical_flow_svg}
            </div>
            """
        )

        gr.HTML(
            """
            <div style="
                width: 100%;
                padding: 2rem;
                border-radius: 12px;
                background: linear-gradient(145deg, #0f172a 0%, #111827 50%, #1e1b4b 100%);
                border: 1px solid #374151;
                box-shadow: 0 12px 25px rgba(0,0,0,0.35);
                margin-bottom: 2rem;
            ">
                
                <h2 style="
                    text-align:center;
                    font-size: 28px;
                    color: #e5e7eb;
                    font-weight: 800;
                    margin-bottom: 1.5rem;
                    letter-spacing: 0.5px;
                ">
                    🚀 How to Use the RAG-Vision Engine
                </h2>

                <div style="
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 1.5rem;
                    padding: 0 1rem;
                ">

                    <!-- STEP 1 -->
                    <div style="
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 14px;
                        padding: 1.5rem;
                        backdrop-filter: blur(6px);
                    ">
                        <h3 style="color:#a5b4fc; margin-bottom:0.8rem;">📂 Step 1: Define Classes</h3>
                        <p style="color:#d1d5db; font-size:15px; line-height:1.6;">
                            Provide two class names (e.g., <b>clean</b> vs <b>dirty</b>) and upload several
                            example images for each class. This builds the <b>support set</b>.
                        </p>
                    </div>

                    <!-- STEP 2 -->
                    <div style="
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 14px;
                        padding: 1.5rem;
                        backdrop-filter: blur(6px);
                    ">
                        <h3 style="color:#a5b4fc; margin-bottom:0.8rem;">✍️ Step 2: Write Prompts</h3>
                        <p style="color:#d1d5db; font-size:15px; line-height:1.6;">
                            Fill in the <b>System Prompt</b> to define the AI persona and output format.
                            Then describe the task in the <b>User Prompt</b>.
                        </p>
                    </div>

                    <!-- STEP 3 -->
                    <div style="
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 14px;
                        padding: 1.5rem;
                        backdrop-filter: blur(6px);
                    ">
                        <h3 style="color:#a5b4fc; margin-bottom:0.8rem;">🖼️ Step 3: Upload Query Image</h3>
                        <p style="color:#d1d5db; font-size:15px; line-height:1.6;">
                            Select the <b>target image</b> you want the AI to classify
                            using retrieval-augmented visual reasoning.
                        </p>
                    </div>

                    <!-- STEP 4 -->
                    <div style="
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 14px;
                        padding: 1.5rem;
                        backdrop-filter: blur(6px);
                    ">
                        <h3 style="color:#a5b4fc; margin-bottom:0.8rem;">🧠 Step 4: Configure Settings</h3>
                        <p style="color:#d1d5db; font-size:15px; line-height:1.6;">
                            Adjust retrieval parameters (<b>k-retrieval</b> and <b>patch size</b>),
                            as well as model generation settings (<b>temperature</b>, <b>top-p</b>).
                        </p>
                    </div>

                    <!-- STEP 5 (Unified) -->
                    <div style="
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 14px;
                        padding: 1.5rem;
                        backdrop-filter: blur(6px);
                    ">
                        <h3 style="color:#a5b4fc; margin-bottom:0.8rem;">🚀 Step 5: Run & Review Output</h3>
                        <p style="color:#d1d5db; font-size:15px; line-height:1.6;">
                            Click <b>Run RAG-Vision Analysis</b> to execute the entire pipeline.
                            <br>
                            Review the predicted <b>binary label</b>, the <b>reasoning</b>,
                            and optionally inspect the <b>raw JSON output</b>.
                        </p>
                    </div>

                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes="glass-panel"):
                gr.Markdown("### 🏷️ Class Definitions & Support Set", elem_classes="section-header")
                
                with gr.Group():
                    with gr.Row():
                        class1_name = gr.Textbox(label="Class 1 Label", placeholder="e.g., dirty")
                        class2_name = gr.Textbox(label="Class 2 Label", placeholder="e.g., clean")
                    
                    imgs_class1 = gr.Gallery(label="Images for Class 1", columns=3, height="auto", interactive=True, elem_classes="custom-gallery")
                    imgs_class2 = gr.Gallery(label="Images for Class 2", columns=3, height="auto", interactive=True, elem_classes="custom-gallery")

            with gr.Column(scale=1, elem_classes="glass-panel"):
                gr.Markdown("### 👁️ Visual Query & Instructions", elem_classes="section-header")
                
                with gr.Group():
                    system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=5,
                    placeholder="Define the AI persona and required JSON output format..."
                    )
                    user_prompt = gr.Textbox(
                    label="User Prompt",
                    lines=5,
                    placeholder="Describe the specific evaluation task..."
                    )

                query_image = gr.Image(label="Target Image", type="filepath", height=None)

        with gr.Accordion("⚙️ Model & Inference Settings", open=False, elem_classes="settings-accordion"):
            with gr.Group(elem_classes="params-panel"):
                gr.Markdown("### 🧠 Generation Parameters")
                with gr.Row():
                    temperature = gr.Slider(0, 1, value=0.2, label="Temperature", info="Controls randomness")
                    top_p = gr.Slider(0, 1, value=0.95, label="Top-p", info="Nucleus sampling threshold")
                    max_new_tokens = gr.Slider(50, 500, value=200, step=10, label="Max Tokens", info="Response length")

                gr.Markdown("### 🔍 Retrieval & Vision Parameters")
                with gr.Row():
                    k_retrieval = gr.Slider(1, 10, value=4, step=1, label="k-Retrieval", info="Support images per class")
                    max_patches_per_class = gr.Slider(1, 10, value=3, step=1, label="Max Patches", info="Visual crops per class")

                with gr.Row():
                    input_resolution = gr.Number(value=224, label="Input Resolution", info="Resize input image")
                    support_res = gr.Number(value=224, label="Support Resolution", info="Resize support images")
                    support_patch_size = gr.Number(value=32, label="Patch Size", info="Patch size for retrieval")

        gr.Markdown("---")
        
        load_example_btn = gr.Button("📦 Load Example Dataset", variant="secondary", size="lg")
        run_button = gr.Button("🚀 Run RAG-Vision Analysis", variant="primary", size="lg")
        reset_btn = gr.Button("🗑️ Reset", variant="stop", size="lg")

        with gr.Group(elem_classes="glass-panel"):
            gr.Markdown("### 📤 Inference Results", elem_classes="section-header")
            with gr.Row():
                result_flag = gr.Textbox(label="Predicted Label", scale=1)
                result_explanation = gr.Textbox(label="Reasoning", lines=1, scale=3)
        
            with gr.Accordion("📄 View Raw JSON Response", open=False, elem_classes="settings-accordion"):
                result_raw = gr.Textbox(
                    label="Raw VLM Output",
                    lines=20,
                    max_lines=999,
                    show_copy_button=True,
                    elem_classes="raw-json"
                )

        current_year = datetime.now().year

        gr.HTML(
            f"""
            <div class="custom-footer">
                © {current_year} — Raul Artigues Femenia · 
                <a href="https://www.raulartigues.com" target="_blank" rel="noopener noreferrer">
                    www.raulartigues.com
                </a>
            </div>
            """
        )

        def run_pipeline(
            class1, class2,
            gallery1, gallery2,
            sys_prompt, usr_prompt,
            q_img,
            temp, top,
            kret, maxpatch, maxtok,
            inpres, supres, suppatch
            ):
            """
            Execute the full RAG-Vision pipeline including:

            1. Validation of class labels and prompts
            2. Uploading support images for both classes
            3. Base64 encoding of the query image
            4. Calling the RAG inference backend service
            5. Returning structured outputs to the UI

            Args:
                class1 (str): Name of the first class.
                class2 (str): Name of the second class.
                gallery1 (list): Images belonging to class 1.
                gallery2 (list): Images belonging to class 2.
                sys_prompt (str): System prompt for the VLM.
                usr_prompt (str): User prompt describing the task.
                q_img (str): Filepath to the query image.
                temp (float): Sampling temperature.
                top (float): Top-p nucleus sampling.
                kret (int): Number of retrieved support images per class.
                maxpatch (int): Max number of visual patches per class.
                maxtok (int): Maximum tokens for generation.
                inpres (int): Input resolution of the query image.
                supres (int): Resolution of support images.
                suppatch (int): Patch size for retrieval.

            Returns:
                tuple:
                    - Predicted flag (str)
                    - Reasoning text (str)
                    - Raw JSON response (str)

            Raises:
                ValueError:
                    If labels or prompts fail validation.
            """
            if not validate_label(class1): return "ERROR", "Invalid Class 1", ""
            if not validate_label(class2): return "ERROR", "Invalid Class 2", ""
            if not validate_english(sys_prompt): return "ERROR", "System Prompt must be English", ""
            
            classes_list = [class1, class2]
            classes_json = json.dumps(classes_list)

            def get_files_from_gallery(gallery_data):
                """
                Extract file paths from a gallery of images.

                Args:
                    gallery_data (list): List of image data, where each item can be
                    a string (single image) or a tuple/list (multiple images).

                Returns:
                    list: List of file paths extracted from the gallery data.
                """
                if not gallery_data: return []
                files = []
                for item in gallery_data:
                    if isinstance(item, (list, tuple)):
                        files.append(item[0])
                    else:
                        files.append(item)
                return files

            files1 = get_files_from_gallery(gallery1)
            files2 = get_files_from_gallery(gallery2)

            def upload_batch(file_list, class_name, start_index):
                """
                Upload a batch of support images for a specific class.

                Args:
                    file_list (list): List of file paths to upload.
                    class_name (str): Name of the class to which the images belong.
                    start_index (int): Starting index for image numbering.

                Returns:
                    tuple:
                        - bool: True if all uploads succeed, False otherwise.
                        - int: Next available index after the upload batch.

                Raises:
                    ValueError:
                        If any image upload fails.
                """
                idx = start_index
                if not file_list: return True, idx
                for fp in file_list:
                    success, err = upload_support_single_image(class_name, classes_json, idx, fp)
                    if not success: return False, f"Error uploading {class_name}: {err}"
                    idx += 1
                return True, idx

            ok, next_idx = upload_batch(files1, class1, 0)
            if not ok: return "ERROR", next_idx, ""
            
            ok, next_idx = upload_batch(files2, class2, next_idx)
            if not ok: return "ERROR", next_idx, ""

            encoded_query = read_file_as_b64(q_img) if q_img else None
            
            output = run_rag_inference(
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                encoded_image=encoded_query,
                temperature=temp,
                top_p=top,
                k_retrieval=kret,
                max_patches_per_class=maxpatch,
                max_new_tokens=maxtok,
                input_resolution=inpres,
                support_res=supres,
                support_patch_size=suppatch
            )
            
            if not output["success"]:
                return "ERROR", output["error"], ""

            return output["flag"], output["explanation"], output["rawResponse"]

        run_button.click(
            fn=run_pipeline,
            inputs=[
                class1_name, class2_name, imgs_class1, imgs_class2,
                system_prompt, user_prompt, query_image,
                temperature, top_p, k_retrieval, max_patches_per_class, max_new_tokens,
                input_resolution, support_res, support_patch_size
            ],
            outputs=[result_flag, result_explanation, result_raw]
        )

        def load_example():
            """
            Load a predefined example dataset into the UI.

            Returns:
                tuple: Pre-filled class names, example images, prompts, and one
                query image suitable for testing the interface.
            """
            ex = build_example_payload()
            return (
                ex["class1"], ex["class2"],
                ex["files_class1"], ex["files_class2"],
                ex["system_prompt"], ex["user_prompt"], ex["query_image"]
            )
        
        load_example_btn.click(
            fn=load_example,
            outputs=[class1_name, class2_name, imgs_class1, imgs_class2, system_prompt, user_prompt, query_image]
        )

        def reset_inputs():
            """
            Reset all user inputs in the interface.

            Returns:
                tuple: Empty strings and None values replacing all user-filled fields.
            """
            return "", "", None, None, "", "", None

        reset_btn.click(
            fn=reset_inputs,
            inputs=[],
            outputs=[class1_name, class2_name, imgs_class1, imgs_class2, system_prompt, user_prompt, query_image]
        )

    return ui
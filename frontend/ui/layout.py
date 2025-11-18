from frontend.services.rag_api_client import upload_support_single_image, run_rag_inference
from frontend.assets.examples.example_payload import build_example_payload
from frontend.utils.text_utils import validate_english, validate_label
from frontend.utils.image_utils import read_file_as_b64
import gradio as gr
import json

def build_ui():
    """
    Build and return the full RAG-Vision Gradio interface.

    This function assembles:
        - The header and project description
        - Class definition inputs
        - Image upload widgets for the support set
        - Prompt fields for the VLM
        - Query image upload
        - Inference parameter controls
        - The pipeline execution logic (upload → inference → display)
        - Output display boxes
        - Example loading buttons
        - Model documentation and footer

    Returns:
        gr.Blocks:
            A fully constructed interactive UI ready for `.launch()`.
    """
    with gr.Blocks(css="""
        footer {visibility: hidden}

        .full-width-header {
            width: 100vw !important;
            margin-left: calc(50% - 50vw) !important;
            padding: 2rem 3rem !important;
        }
        .header-inner {
            max-width: 1200px;
            margin: 0 auto;
            font-size: 0.85rem;
            line-height: 1.4;
            text-align: justify;
        }

        .header-inner h2 {
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }

        .header-inner h3 {
            font-size: 0.85rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 768px) {
            .header-inner { font-size: 0.85rem; }
        }
    """) as ui:

        gr.HTML(
            """
        <div class="full-width-header">
            <div class="header-inner">

                <h2>🔍 <b>RAG-VISION ENGINE — Visual Reasoning Interface</b></h2>

                <p>
                This interface provides a <b>semi-automatic labeling system</b> powered by 
                <b>retrieval-augmented vision models (RAG-Vision)</b>. It allows you to upload two 
                support classes (e.g., <i>dirty</i> vs <i>clean</i> cars), define English prompts, 
                and run visual reasoning over a new image.
                </p>

                <p>
                The goal is to evaluate the feasibility of RAG-Vision for tasks such as 
                <b>automatic data labeling</b>, <b>few-shot visual classification</b>, and 
                <b>retrieval-enhanced visual reasoning</b>.
                </p>
                <h3><b>Author</b></h3>
                <p><b>Raul Artigues Femenia</b></p>
                <p>Project Repository: 
                    <a href="https://github.com/raulartigues/rag-vision-engine" 
                    target="_blank" rel="noopener noreferrer">
                        https://github.com/raulartigues/rag-vision-engine
                    </a>
                </p>
                <p>License: MIT</p>

            </div>
        </div>
            """
        )

        gr.Markdown("---")

        gr.Markdown("## 🏷️ Define Your Two Support Classes")

        load_example_btn = gr.Button("📦 Load Example", variant="secondary")

        with gr.Row():
            class1_name = gr.Textbox(
                label="Class 1 Name (English)",
                placeholder="e.g., dirty",
                max_lines=1,
                container=True,
            )
            class2_name = gr.Textbox(
                label="Class 2 Name (English)",
                placeholder="e.g., clean",
                max_lines=1,
                container=True,
            )

        gr.Markdown("---")

        gr.Markdown("## 🖼️ Support Images")

        with gr.Row():
            imgs_class1 = gr.File(
                label="Upload Images for Class 1",
                file_count="multiple",
                file_types=["image"],
            )
            imgs_class2 = gr.File(
                label="Upload Images for Class 2",
                file_count="multiple",
                file_types=["image"],
            )

        with gr.Row():
            preview1 = gr.Gallery(
                label="Class 1 Preview",
                columns=1,
                height=250,
                show_label=True,
                preview=True,
            )
            preview2 = gr.Gallery(
                label="Class 2 Preview",
                columns=1,
                height=250,
                show_label=True,
                preview=True,
            )

        imgs_class1.change(lambda f: f, imgs_class1, preview1)
        imgs_class2.change(lambda f: f, imgs_class2, preview2)

        gr.Markdown("---")

        gr.Markdown("## 📜 Prompts & Query Image")

        system_prompt = gr.Textbox(
            label="System Prompt",
            lines=4,
            placeholder="Write the system instructions here...",
        )

        user_prompt = gr.Textbox(
            label="User Prompt",
            lines=4,
            placeholder="Write the user instructions here...",
        )

        query_image = gr.Image(
            label="Query Image",
            type="filepath",
            height=300
        )

        gr.Markdown("---")

        gr.Markdown("## ⚙️ Inference Parameters")

        with gr.Row():
            temperature = gr.Slider(0, 1, value=0.2, label="Temperature")
            top_p = gr.Slider(0, 1, value=0.95, label="Top-p")

        gr.Markdown(
            """
            <div style='font-size: 0.85rem; line-height: 1.4;'>
            🧠 <b>Temperature</b>: Controls randomness. Lower = more deterministic, higher = more diverse.<br>
            🎯 <b>Top-p</b>: Nucleus sampling. Limits token selection to the most probable subset.
            </div>
            """,
        )

        with gr.Row():
            k_retrieval = gr.Slider(1, 10, value=4, step=1, label="k-Retrieval")
            max_patches_per_class = gr.Slider(
                1, 10, value=3, step=1, label="Max Patches Per Class"
            )
            max_new_tokens = gr.Slider(
                50, 500, value=200, step=10, label="Max New Tokens"
            )

        gr.Markdown(
            """
            <div style='font-size: 0.85rem; line-height: 1.4;'>
            🔍 <b>k-Retrieval</b>: Number of support images used for retrieval.<br>
            🧩 <b>Max Patches Per Class</b>: Maximum number of visual patches extracted per class.<br>
            ✏️ <b>Max New Tokens</b>: Maximum output length generated by the model.
            </div>
            """,
        )

        with gr.Row():
            input_resolution = gr.Number(value=224, label="Input Resolution")
            support_res = gr.Number(value=224, label="Support Resolution")
            support_patch_size = gr.Number(value=32, label="Patch Size")

        gr.Markdown(
            """
            <div style='font-size: 0.85rem; line-height: 1.4;'>
            📸 <b>Input Resolution</b>: Resolution applied to the query image.<br>
            🖼️ <b>Support Resolution</b>: Resolution used for support images.<br>
            🧱 <b>Patch Size</b>: Size (in pixels) of visual patches taken from images.
            </div>
            """,
        )

        gr.Markdown("---")

        run_button = gr.Button("🚀 Run RAG-Vision", variant="primary")

        gr.Markdown("## 📤 Output")

        result_flag = gr.Textbox(label="Flag")
        result_explanation = gr.Textbox(label="Explanation", lines=5)
        result_raw = gr.Textbox(label="Raw VLM Output", lines=10)

        def run_pipeline(
            class1, class2,
            files1, files2,
            sys_prompt, usr_prompt,
            q_img,
            temp, top,
            kret, maxpatch, maxtok,
            inpres, supres, suppatch
            ):
            """
            Pipeline executed when clicking the "Run RAG-Vision" button.

            Steps:
                1. Validate class labels and prompt language.
                2. Build class list and JSON encoding.
                3. Upload support images sequentially to backend.
                4. Base64 encode query image.
                5. Call backend inference endpoint.
                6. Return results back into UI elements.

            Returns:
                tuple(str, str, str):
                    (flag, explanation, raw_response)
                    Or ("ERROR", error_message, "") on failure.
            """
            if not validate_label(class1):
                return "ERROR", "Class 1 is invalid (must be simple English label)", ""
            if not validate_label(class2):
                return "ERROR", "Class 2 is invalid (must be simple English label)", ""

            if not validate_english(sys_prompt):
                return "ERROR", "System Prompt must be in English", ""
            if not validate_english(usr_prompt):
                return "ERROR", "User Prompt must be in English", ""

            classes_list = [class1, class2]
            classes_json = json.dumps(classes_list)

            global_index = 0

            def upload_batch(file_list, class_name, start_index):
                idx = start_index
                for fp in file_list:
                    success, err = upload_support_single_image(
                        className=class_name,
                        classes_json=classes_json,
                        index=idx,
                        filepath=fp
                    )
                    if not success:
                        return False, f"Upload failed at {class_name} #{idx}: {err}"
                    idx += 1

                return True, idx

            ok, next_index = upload_batch(files1, class1, global_index)
            if not ok:
                return "ERROR", next_index, ""

            ok, next_index = upload_batch(files2, class2, next_index)
            if not ok:
                return "ERROR", next_index, ""

            print(f"Support upload complete. Total: {next_index} images")

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
                support_patch_size=suppatch,
                flag="VisibleDirtyFlag"
            )
            
            if not output["success"]:
                return "ERROR", output["error"], ""

            return (
                output["flag"],
                output["explanation"],
                output["rawResponse"]
            )

        run_button.click(
            fn=run_pipeline,
            inputs=[
                class1_name, class2_name,
                imgs_class1, imgs_class2,
                system_prompt, user_prompt,
                query_image,
                temperature, top_p,
                k_retrieval, max_patches_per_class, max_new_tokens,
                input_resolution, support_res, support_patch_size
            ],
            outputs=[result_flag, result_explanation, result_raw]
        )

        gr.Markdown("---")
        gr.HTML(
            """
            <div class="header-inner">

                <h3><b>Models Used</b></h3>

                <h4>🔗 Qwen2-VL-2B (Vision-Language Model)</h4>
                <ul>
                    <li>GitHub: <a href="https://github.com/QwenLM/Qwen2-VL" target="_blank">https://github.com/QwenLM/Qwen2-VL</a></li>
                    <li>Paper: <a href="https://arxiv.org/abs/2404.11156" target="_blank">https://arxiv.org/abs/2404.11156</a></li>
                    <li>License: Apache 2.0</li>
                </ul>

                <h4>🔗 CLIP ViT-B/32 (Visual Encoder)</h4>
                <ul>
                    <li>GitHub: <a href="https://github.com/openai/CLIP" target="_blank">https://github.com/openai/CLIP</a></li>
                    <li>Paper: <a href="https://arxiv.org/abs/2103.00020" target="_blank">https://arxiv.org/abs/2103.00020</a></li>
                    <li>License: MIT</li>
                </ul>

            </div>
            """
        )

        def load_example():
            """
            Load predefined example data:
                - Example class names
                - Example support images
                - Example prompts
                - Example query image

            Returns:
                tuple:
                    Values to populate form fields directly.
            """
            example = build_example_payload()

            class1 = example["class1"]
            class2 = example["class2"]

            files1 = example["files_class1"]
            files2 = example["files_class2"]

            sys_p = example["system_prompt"]
            usr_p = example["user_prompt"]

            q_img = example["query_image"]

            return (
                class1,
                class2,
                files1,
                files2,
                sys_p,
                usr_p,
                q_img
            )
        
        load_example_btn.click(
            fn=load_example,
            inputs=[],
            outputs=[
                class1_name,
                class2_name,
                imgs_class1,
                imgs_class2,
                system_prompt,
                user_prompt,
                query_image
            ]
        )

        gr.HTML(
            """
            <div style="
                text-align: center;
                width: 100%;
                margin-top: 3rem;
                margin-bottom: 1rem;
                color: #888;
                font-size: 0.8rem;
            ">
                © 2025 — Raul Artigues Femenia · 
                <a href="https://www.raulartigues.com" 
                   target="_blank" 
                   rel="noopener noreferrer"
                   style="color: #88aaff;">
                   www.raulartigues.com
                </a>
            </div>
            """
        )

    return ui
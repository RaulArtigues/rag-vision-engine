import textwrap

class APISettings:
    app_name: str = "RAG VISUAL API"
    app_name_underscore: str = app_name.replace(' ', '_')
    app_name_identifier: str = "rag_vision_engine"
    api_current_version: str = "1.0.0"
    app_description: str = textwrap.dedent("""
RAG Vision Engine API provides a Retrieval-Augmented Visual Reasoning system 
designed for few-shot image classification, multimodal explanation generation, 
and retrieval-enhanced vision-language inference.

This API combines two major components:

1. **CLIP ViT-B/32**  
   Used for patch-level embedding and retrieval across a support image dataset.  
   The system extracts patches from support images and query images, 
   computes CLIP embeddings, and selects the most similar patches as evidence.

2. **Qwen2-VL-2B-Instruct**  
   A compact yet powerful vision-language model used for multimodal reasoning.  
   It receives the query image + retrieved patches along with system and user prompts, 
   and produces:
   - A binary classification flag  
   - An explanatory reasoning text  
   - Class similarity scores  
   - A full decoded VLM response

---

## 🔍 Core Features

- Patch-based retrieval using CLIP  
- Few-shot classification using support images  
- Multimodal VLM-based reasoning  
- Support class upload and management  
- Fully integrated FastAPI backend & Gradio UI  
- JSON-based remote inference for programmatic usage  

---

## 🧩 Main Endpoints

### **POST /ragvision/invocations**
Runs the full RAG-Vision inference pipeline:
- Decodes the query image
- Retrieves patch-level evidence using CLIP
- Sends all context to Qwen2-VL
- Returns a structured response including flag, explanation, raw response and scores

### **POST /support/upload/image**
Uploads support images for each class (e.g., "dirty", "clean").
Used to build the retrieval index dynamically.

### **GET /ragvision/healthcheck**
Provides service health status and readiness checks.

---

## 🧠 Intended Use

This API is intended for:
- Semi-automatic dataset labeling  
- Visual quality inspection  
- Few-shot visual classification tasks  
- Vision-language prototyping  
- Retrieval-augmented AI research  

---

## 📦 Response Format

Most inference responses follow the `RagVisionOutput` schema:
- `success`: Boolean status  
- `flag`: Binary classification output  
- `explanation`: VLM-generated reasoning  
- `classScores`: Aggregated similarity scores per class  
- `rawResponse`: Complete LLM/VLM output  
- `imageId`: Unique identifier for the inference request  

---

## 📘 Notes

- All inference runs are stateless aside from support image uploads.  
- Support data is stored per-session under `backend/data/support/`.  
- The system supports CPU or GPU execution depending on environment.  
- Qwen2-VL and CLIP are loaded locally for optimal performance.

""")
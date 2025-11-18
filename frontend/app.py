from frontend.ui.layout import build_ui

if __name__ == "__main__":
    """
    Entry point for launching the RAG-Vision frontend application.

    This script:
        - Builds the UI using `build_ui()` from the layout module.
        - Launches the frontend server (typically a Gradio interface).
        - Exposes the app publicly if `share=True` is enabled.

    Notes:
        - `server_name="0.0.0.0"` allows external access, making the UI reachable
          inside containers or cloud environments.
        - `server_port=None` lets the framework auto-select an available port.
        - `show_api=False` hides extra API documentation panels when using Gradio.
        - `share=True` is required for public Gradio links (e.g., for demos or Spaces).
    """
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=None,
        show_api=False,
        share=True
    )
"""ASGI entrypoint. Run with: ``uvicorn app.serve:app --host 0.0.0.0 --port 7860``.

Using a module entrypoint (not ``python app/gradio_app.py``) keeps the module
identity stable so the JSON API and the Gradio handlers share one model cache.
"""

from app.gradio_app import build_app

app = build_app()


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

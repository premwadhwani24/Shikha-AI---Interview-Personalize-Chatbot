import os, base64, requests

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")

# Generates a 4K image using Stability API (or any compatible diffusion server)
# Falls back to a placeholder if key missing.

def generate_4k(prompt: str) -> str:
    if not STABILITY_API_KEY:
        # return tiny transparent PNG (base64) as fallback
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9a0b7n0AAAAASUVORK5CYII="
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    data = {
        "prompt": prompt,
        "output_format": "png",
        "aspect_ratio": "16:9",
        "width": 3840,
        "height": 2160,
        "cfg_scale": 7,
        "steps": 30,
    }
    r = requests.post(url, headers=headers, files={"none": ("none", "")}, data={k: str(v) for k,v in data.items()}, timeout=120)
    r.raise_for_status()
    return "data:image/png;base64," + base64.b64encode(r.content).decode()
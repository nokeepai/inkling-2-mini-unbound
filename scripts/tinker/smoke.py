"""Tinker smoke: check the key works and the free Mini model is listed. Spends nothing.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/smoke.py
"""
import os
from pathlib import Path

os.environ["TINKER_API_KEY"] = Path("~/.inkling-abl/tinker_key").expanduser().read_text().strip()
import tinker  # noqa: E402

FREE = "thinkingmachines/Inkling-2-Mini:peft:65536:free"

caps = tinker.ServiceClient().get_server_capabilities()
names = [m.model_name for m in caps.supported_models]
print("models:", len(names))
for n in names:
    if "Inkling" in n:
        print(" ", n)
print("free model listed:", FREE in names)

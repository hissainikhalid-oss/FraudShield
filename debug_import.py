# debug_import.py
import traceback

try:
    import detector.views as v
    print("Imported module:", v.__file__)
    print("Available ai/api names:", [n for n in dir(v) if n.startswith('ai') or n.startswith('api_')])
except Exception:
    traceback.print_exc()
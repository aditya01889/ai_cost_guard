# test_imports.py
import sys
print("Python path:", sys.path)

try:
    import ai_cost_guard
    print("✅ ai_cost_guard imported successfully")
    print("Module location:", ai_cost_guard.__file__)
except ImportError as e:
    print("❌ Failed to import ai_cost_guard:", e)

try:
    from ai_cost_guard.core.guardrails import GuardrailConfig
    print("✅ GuardrailConfig imported successfully")
except ImportError as e:
    print("❌ Failed to import GuardrailConfig:", e)
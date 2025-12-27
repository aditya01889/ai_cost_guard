# ai_cost_guard/demo/seed_demo_data.py

from datetime import datetime
from ai_cost_guard.storage.repository import initialize_schema, insert_usage_event
from ai_cost_guard.storage.models import LLMUsageEvent

initialize_schema()

events = [
    LLMUsageEvent(
        timestamp=datetime.now(),
        feature="document_summary",
        model="gpt-3.5-turbo",
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        estimated_cost=3.20,
        retry_count=1
    ),
    LLMUsageEvent(
        timestamp=datetime.now(),
        feature="document_summary",
        model="gpt-3.5-turbo",
        prompt_tokens=4000,
        completion_tokens=1000,
        total_tokens=5000,
        estimated_cost=9.10,  # spike
        retry_count=1
    )
]

for e in events:
    insert_usage_event(e)

print("Demo usage data inserted")

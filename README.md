# AI Cost Guard

**Deterministic AI Cost Governance for Production Systems**

AI Cost Guard is a **developer-first safety layer** that prevents runaway AI spend **before it hits your bill**.

It works across:

* Runtime execution
* Guardrails & kill-switches
* CI/CD simulation

No heuristics.
No black boxes.
No surprise bills.

## Demo

Watch the demo video: [AI Cost Guard Demo](https://www.loom.com/share/f6caedbd6e694f29a6f8b61e1cfd61ca)

---

## Why AI Cost Guard Exists

AI-assisted development introduces **new failure modes** that traditional monitoring does not catch:

* A prompt change doubles token usage
* A retry loop multiplies cost silently
* A deploy shifts traffic patterns overnight
* CI approves code that burns money in prod

Most teams discover these **after the invoice arrives**.

AI Cost Guard stops this **before it ships**.

---

## What This Tool Does

AI Cost Guard provides **end-to-end cost governance** for LLM usage:

### 1. Immutable Cost Ledger

Every LLM call records:

* Tokens
* Estimated cost
* Feature + model
* Retry behavior

This data is **append-only** and auditable.

---

### 2. Deterministic Baselines

The system learns **normal cost behavior** per feature/model:

* Median cost
* P90 cost
* Median tokens
* Cold vs warm baseline state

No ML. No guessing. Fully explainable.

---

### 3. Conservative Anomaly Detection

Detects:

* Cost spikes
* Token explosions
* Retry amplification

Each anomaly includes:

* Observed value
* Baseline reference
* Threshold
* Human-readable explanation

---

### 4. Guardrail Enforcement (Kill-Switch)

Explicit policies let you:

* Block expensive requests
* Throttle runaway features
* Downgrade models
* Warn without blocking

Enforcement is:

* Deterministic
* Auditable
* Ordered safely (hard limits → budgets → anomalies)

---

### 5. CI-Safe Cost Simulation

Before merging code, simulate:

> “If this ships, will AI costs violate guardrails?”

* Read-only
* Uses the exact same logic as runtime
* Can block deploys automatically

---

## Quick Start (Demo Mode — No API Keys Needed)

You **do not need a real OpenAI API key** to try AI Cost Guard.

### 1. Install & activate environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

---

### 2. Seed demo usage data

```bash
python ai_cost_guard/demo/seed_demo_data.py
```

This inserts:

* Normal usage
* Cost spikes
* Retry amplification

No external APIs required.

---

### 3. Run simulation

```bash
python -m ai_cost_guard.cli.main simulate
# Or, after installation:
ai-cost-guard simulate
```

Example output:

```
AI Cost Simulation Result
----------------------------------------

Feature: document_summary
Baseline cost/request: $6.15
Simulated cost/request: $12.30
Change: +100.0%
Estimated monthly impact: $12.30
```

Exit code:

* `0` → PASS or WARN
* `1` → FAIL (CI-blocking)

---

## CI Integration Example

```yaml
- name: AI Cost Simulation
  run: python -m ai_cost_guard.cli.main simulate --enforced
```

If cost guardrails are violated:

* CI fails
* Deploy is blocked
* No money is burned

---

## Design Principles (Non-Negotiable)

AI Cost Guard is intentionally **boring and strict**:

* ❌ No ML
* ❌ No auto-healing
* ❌ No hidden thresholds
* ❌ No silent fallbacks

Everything is:

* Deterministic
* Explicit
* Reviewable
* Testable

If an engineer cannot explain *why* something was blocked, the design has failed.

---

## Who This Is For

* Teams shipping AI features weekly
* Startups watching burn closely
* Enterprises needing auditability
* Anyone who has said:
  **“How did our AI bill get this high?”**

---

## What This Is Not

* Not an observability dashboard
* Not a cost optimizer
* Not a billing predictor
* Not a usage analytics toy

AI Cost Guard is **infrastructure**, not a report.

---

## Roadmap (Short-Term)

* CI status command (`ai-cost-guard status`)
* Guardrail violation reports
* Baseline visualization
* Multi-provider adapters

---

## Philosophy

> AI systems fail silently.
> Money leaks quietly.
> Guardrails must be explicit.

AI Cost Guard exists so your **success doesn’t depend on luck**.

---

## License

MIT



# `gemma_active/gemma_revision_audit.py`

## Status

Active helper/audit script.

## Purpose

Audit the local Gemma setup for possible runtime or revision drift.

Use this when you suspect that the model/tokenizer/runtime changed, or when a
new run behaves differently from previous runs.

## What It Is For

This script is not a steering experiment. It is a diagnostic helper for
checking whether the environment appears consistent with the intended Gemma
setup.

## Typical Use Cases

- You suspect the hosted model revision changed.
- You suspect tokenizer or chat-template behavior changed.
- You want a lightweight record of what model/config/runtime was used.
- You want to compare a new Colab session against old runs.

## Typical Usage

```python
%run -i gemma_revision_audit.py
```

Local project path:

```python
%run -i experiments/steering/sae_gemma_qwen/gemma_active/gemma_revision_audit.py
```

## Interpretation Role

Use this as an audit trail tool, not as evidence for the hidden-state shift by
itself.



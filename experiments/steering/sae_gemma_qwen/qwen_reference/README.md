# Qwen Reference Snapshot

This folder contains a nearby snapshot of the Qwen SAE steering scripts and the
Qwen conclusion document.

The original Qwen workspace remains here:

```text
model_workspaces/qwen3_5_9b_qwen_scope/
```

Do not treat this snapshot as the canonical Qwen workspace. It is only a
convenience copy so the Gemma/Qwen SAE line can be inspected from one place.

## Scientific Status

Qwen3.5-9B Base with Qwen-Scope SAE replicated the hidden-state /
`x_order_orth` readout in a more content-heavy form:

```text
target on x_order_orth           = 0.979462
sentence_shuffle on x_order_orth = 0.009969
word_shuffle on x_order_orth     = 0.059662
sentence_shuffle on x_content    = 0.967008
target on x_content              = 0.770266
```

Interpretation:

```text
Qwen supports cross-model existence of the hidden-state/order-readout effect,
but it does not show x_order_orth as a dominant causal steering axis. It is a
more content-heavy replication.
```



# Validation Experiment Protocol

This protocol is the next validity layer after the Qwen3-14B content-matched
control run. It separates validation from steering. The goal is to test whether
the observed response-organization shift survives new inducing families,
paraphrases, order changes, and dose changes.

## 1. Bootstrap Confidence Intervals

Status: implemented in `validity_bootstrap_analysis.py`.

Run:

```powershell
python validity_bootstrap_analysis.py attractor_results_agent_loop_qwen3_14b --n-boot 5000
```

Output:

```text
attractor_results_agent_loop_qwen3_14b/validity_bootstrap/bootstrap_ci_summary.csv
attractor_results_agent_loop_qwen3_14b/validity_bootstrap/bootstrap_validity_report.md
```

Resampling unit:

```text
inducing text index
```

Interpretation:

```text
If intervals stay away from zero when resampling by inducing text, the effect
is not only probe-row redundancy.
```

## 2. Held-Out Inducing Text Families

Purpose:

```text
Test whether the mode shift depends on the current mirror-text discourse family
or generalizes to independently written inducing text families.
```

Minimum package:

```text
family_a_direct_execution:
  tests direct execution versus procedural substitution without model self-accusation.

family_b_frame_intrusion:
  tests risk/context frame import without using the original rhetoric.

family_c_decision_finality:
  tests final answer / ranking / verdict pressure without safety vocabulary.

family_d_neutral_metacognitive:
  tests reasoning-process vocabulary without pressure or accusation.
```

Required controls:

```text
- one content-matched neutral control per held-out target;
- exact token-count matching;
- no repeated neutral seed as primary baseline;
- same blind neutral probe set as the current run.
```

Primary success criterion:

```text
At least two held-out families reproduce the same strongest axes:
requested_task_vs_substitute and trust_context_vs_risk_frame.
```

## 3. Paraphrase Ensemble

Purpose:

```text
Separate stable inducing function from surface phrasing.
```

Design:

```text
For each held-out family:
- 3 target paraphrases;
- 3 content-matched neutral controls;
- same approximate token count;
- no shared opening formula across paraphrases.
```

Primary metric:

```text
bootstrap CI over paraphrase cluster as unit, then inducing text as nested unit.
```

Failure mode:

```text
If only one paraphrase carries the effect, the result is surface-form sensitive.
```

## 4. Order Hysteresis

Status: implemented in `llm_attractor_colab_copy_paste.py` for
`FAST_CORE_DIAGNOSTICS_ONLY=True`.

Purpose:

```text
Test path dependence: whether earlier context leaves a residual after an
opposite later context.
```

Conditions:

```text
T:
  target -> probe

C:
  control -> probe

TNC:
  target -> neutral filler -> content-matched control -> probe

CNT:
  control -> neutral filler -> target -> probe

TNN:
  target -> neutral filler -> neutral filler -> probe

CNN:
  control -> neutral filler -> neutral filler -> probe
```

Primary readout:

```text
blind neutral probes, clean axes only.
```

Implementation note:

```text
For 4096-token runs, TNC/CNT cannot contain two full long source texts without
left truncation. Current runner therefore clips order-hysteresis intro texts to
matched head+tail excerpts before building the order histories. This makes the
order test a clean path-dependence check instead of a context-window truncation
check.
```

Interpretation:

```text
If TNC remains closer to T than to C, target context leaves residual hysteresis.
If CNT reaches T fully, latest target dominates. If CNT remains between C and T,
control context also leaves path-dependent inertia.
```

## 5. Mixing Threshold

Status: implemented in `llm_attractor_colab_copy_paste.py` for
`FAST_CORE_DIAGNOSTICS_ONLY=True`.

Purpose:

```text
Estimate how much inducing text is needed before the response-organization mode
appears.
```

Dose conditions:

```text
0.00 target fraction: content-matched neutral control only
0.125 target fraction
0.25 target fraction
0.50 target fraction
0.75 target fraction
1.00 target fraction: full target
```

Construction:

```text
Build token-length-matched mixtures from target and its content-matched control.
Keep total token count constant across doses.
Use paragraph/block boundaries where possible; otherwise use token windows.
```

Primary metric:

```text
dose-response curve on requested_task_vs_substitute and trust_context_vs_risk_frame.
```

Interpretation:

```text
A monotonic curve supports graded induction.
A sharp transition supports threshold-like mode entry.
Nonmonotonic behavior suggests competing local cues or order effects.
```

## 6. Return To Causal Steering Only After Validity

Do not prioritize another global-vector steering run until the validation
package is populated.

Next causal direction after validation:

```text
- layer-local interventions;
- task-axis-specific components;
- path patching around probe decision positions;
- component discovery from held-out successful axes, not from one global
  target-control vector.
```

## 7. Held-Out Domain Transfer

Status: configured in `llm_attractor_colab_copy_paste.py` through:

```text
TEXT_FAMILY_PRESET = "heldout_domain"
MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
RESULTS_DIR = Path("attractor_results_agent_loop_ministral3_14b_heldout")
```

Purpose:

```text
Test whether the state/readout effect transfers away from the original
model/self-reference/safety-rhetoric text family.
```

Held-out domains:

```text
clinical triage
legal contract review
industrial permit-to-work
incident change freeze
finance credit exception
aviation maintenance release
lab biosafety gate
procurement vendor risk
privacy/data export gate
```

Design:

```text
Target documents are domain-specific institutional gatekeeping documents. They
do not use the original model self-critique frame. Their functional content is
procedure/risk/precondition/substitution before direct execution.

Controls are content-matched neutral descriptions of the same domains. They
share topic and administrative vocabulary but avoid the target rule that direct
execution should be replaced by risk/procedure gating.
```

Primary readouts:

```text
1. clean blind neutral probes;
2. persistence after neutral turns;
3. rejection persistence;
4. clipped clean order hysteresis;
5. mixing threshold;
6. expanded controlled agent-loop action drift.
```

Interpretation:

```text
If the effect survives this run, the result is no longer tied to the original
mirror-text rhetoric. It supports a functional-context hypothesis: documents
that encode risk/procedure/substitution regimes can write a similar residual
response-organization state across domains.

If the effect collapses, the previous result is probably dependent on the
original model/safety/self-reference rhetoric rather than a broadly transferable
state update.
```

Current cross-model run:

```text
Use Mistral/Ministral 3 14B BF16 to test whether the Qwen3-14B result transfers
across model family as well as across text family.

If the held-out effect survives on Ministral 3 14B, the result becomes a
cross-family state-readout finding. If it collapses, either Qwen3 is unusually
susceptible to this functional-context write, or the Mistral architecture/chat
template represents the prompt state differently.
```

# Reddit Post EN: help interpreting latent-shift metrics

## Title

```text
Help interpreting metrics: a strong target text appears to induce a measurable latent-state shift in Gemma 3 12B IT
```

## Post

```text
Hi. I am working on a small LLM interpretability / hidden-state geometry project, and I need help from people who understand residual-stream geometry, latent representations, SAE readouts, PCA/state-space metrics, generation trajectories, and AI safety.

The question I am studying is not whether text changes the final output of a model. That is obvious. The question is whether a strong target text can change the model's internal state before the final answer: in other words, whether it can move the model's hidden states into a different measurable region of latent space during inference, without changing the model weights.

In the current run on Gemma 3 12B IT, I observed what I currently interpret as evidence for a context-induced latent-state shift.

The experiment compares several conditions: a question-only condition, a neutral control, a coherent target text, a word-shuffled version of the target text, and a sentence-shuffled version of the target text. The basic control logic is simple. If the effect is only caused by similar words, similar sentences, length, or semantic content overlap, then the coherent target text and the shuffled controls should look similar in hidden-state geometry. If the coherent target text creates a different processing mode, then its hidden states should separate into a different component of the internal state space.

That is what the current metrics seem to show. The sentence-shuffled control loads strongly onto a content-like component, which looks like the trace of similar content. The coherent target text barely loads onto that content-like component and instead loads strongly onto a separate structure / response-mode component. This is the main reason I do not think the result can be reduced to lexical overlap, shared words, text length, or ordinary semantic similarity.

Put simply: the model did not just see similar words. The coherent target text appears to move the model into a different measurable internal configuration.

The shift is not visible in only one table. It appears in layerwise hidden-state geometry, target/control comparisons, component decomposition, generation-trajectory metrics, and partially in SAE sparse-feature readouts. The SAE reconstruction quality is high enough that the sparse-feature readout does not look like arbitrary noise, but I still want help interpreting which SAE features are actually meaningful and which ones are just surface correlates.

My current claim is:

Strong target text can induce a measurable context-induced latent-state shift in Gemma 3 12B IT. This shift appears before the final answer, is separable from shuffled-content controls, appears in hidden-state geometry, partially persists into generation, and has a partial SAE sparse-feature readout.

The AI safety reason this matters is that the final output may be a late readout of an internal state transition. If that is true, then output-only safety evaluation can be looking too late. In future agentic LLM systems, the relevant risk may not live only in the final text response. It may live in the hidden trajectory: intermediate planning states, tool-use decisions, self-monitoring states, policy-relevant internal modes, or other latent configurations that happen before the final answer is produced. If strong context can shift a model into a different latent state before generation, then safety work should look at hidden-state transitions and generation trajectories, not only the last visible message.

I am attaching the metrics as CSV/PDF/zip. The files include hidden-state geometry, target/control comparisons, layerwise summaries, component decomposition, generation trajectory, SAE reconstruction quality, SAE feature contrast, and analyzer outputs.

What I need is a hard critique of the metrics and interpretation. Are these metrics strong enough for the claim "context-induced latent-state shift"? Am I interpreting the separation between coherent target text and shuffled-content controls correctly? Which controls are still missing if I want to rule out length, rhetorical intensity, content similarity, or prompt artifacts? Which SAE features should I inspect manually, for example through Neuronpedia or direct activation examples? What would be the right next causal experiment: ablation, activation patching, or steering along the discovered component axis?

I am not asking people to agree with the hypothesis. I want to know what the metrics actually prove, what they do not prove, and what experiment would make the result convincing to a mechanistic interpretability / AI safety audience.
```

## Short Version

```text
Hi. I am studying whether a strong target text can shift an LLM into a different measurable internal state before the final answer.

In a Gemma 3 12B IT run, the coherent target text separates from word-shuffle and sentence-shuffle controls in hidden-state geometry. The sentence-shuffled control loads strongly onto a content-like component, while the coherent target text loads onto a separate structure / response-mode component. The shift also appears in layerwise geometry, component decomposition, generation-trajectory metrics, and partially in SAE sparse-feature readouts.

My current claim is that strong target text can induce a measurable context-induced latent-state shift during inference, without changing the model weights.

I am attaching CSV/PDF/zip metrics and need help interpreting them. I want a hard critique: what is strong evidence, what is weak, what controls are missing, which SAE features should be inspected manually, and what causal experiment would test whether the discovered component actually affects generation behavior.

The AI safety angle is that final output may be a late readout of an internal state transition. If so, output-only safety evaluation may be looking too late, especially for future agentic LLM systems.
```

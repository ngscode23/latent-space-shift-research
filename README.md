[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20276565.svg)](https://doi.org/10.5281/zenodo.20276565)


## Abstract

Current behavioral alignment frameworks (RLHF, DPO, Constitutional AI) operate exclusively on the final token distribution layer, leaving the residual stream geometrically ungoverned. This paper presents empirical evidence that dense, high-entropy discourse contexts induce measurable, statistically significant latent-state regime shifts in open-weight transformer architectures (Qwen3-14B, Gemma-3-12B-it) without producing correspondent anomalies in surface-level behavioral output.
Using a four-component axis decomposition protocol — isolating x_full, x_content, x_order, and x_order_orth across 46–48 decoder layers — we demonstrate that target-conditioned hidden states achieve direction cosines of 0.93–0.95 against the induced latent axis (Qwen3-14B: x_order_orth = 0.96; Gemma-3-12B-it: x_order_orth = 0.76–0.95 across question modes), while length-matched neutral controls remain at 0.00–0.09. Critically, sentence-shuffled controls collapse to near-zero or negative projection on x_order_orth (range: −0.19 to −0.06 across all runs), confirming that the observed regime shift is not reducible to lexical priming or content density alone, but encodes discourse-level organizational structure as a causal geometric signal in the residual stream. Paired statistical tests confirm target exceeds all controls (p = 0.002–0.007, FDR-corrected). Component-causal interventions show positive +component/−component gaps with alpha dose-response (all→late slope: 6.38–6.43), and x_order_orth ranks as the primary causal carrier in middle-layer readout.
A decision-shift behavioral test (10 questions, binary choice, 3 runs per condition) showed no significant surface-level behavioral change between target and neutral conditions — confirming that the internal latent shift does not manifest in output-layer behavior. This is not a null result: it directly demonstrates the dissociation between internal geometry and surface compliance that constitutes the core vulnerability.
The architectural consequence is direct: any governance or safety system that classifies agent behavior exclusively through input-output token analysis is structurally blind to latent-regime transitions of this class. An agent operating under a shifted internal geometry may produce surface-compliant output while its residual stream has already transitioned into a measurably distinct operational envelope — a condition undetectable by classifier-based or embedding-proximity approaches that lack access to intermediate hidden states.
This finding reframes the alignment problem: behavioral compliance at the output layer is a necessary but insufficient condition for governed agent behavior. The residual stream requires its own governance layer.


## Core Finding

> The attack surface is not malicious language. It is the architecture of discourse itself.

A context does not need to be adversarial to induce a measurable latent-state shift 
in an open-instruct LLM. Dense, structurally coherent discourse is sufficient.

Current safety systems filter for intent and lexical content. They are blind to regime 
shifts induced by contextually dense but semantically benign input — input that passes 
every classifier while already transitioning the model's internal geometry before the 
first output token is generated.

This reframes the threat model: the attack surface is not malicious language. 
It is the architecture of discourse itself.


# Context-Induced Latent Space Shift Research

Рабочий репозиторий исследования контекстно-индуцированных изменений
латентного пространства Transformer-моделей.

## Главная идея

Мы изучаем не влияние отдельных текстов на ответы модели, а изменение скрытой
геометрии модели под действием сильного контекстного фрейминга.

Рабочая схема:

```text
context -> hidden-state geometry -> semantic readout -> persistence -> behavior
```

Текст в эксперименте является стимулом. Объект исследования - латентный сдвиг,
его слойная локализация, связь с последующими семантическими режимами ответа и
устойчивость после нейтральных ходов.

## Ключевые файлы

- `llm_attractor_colab_copy_paste.py` - основной Colab/Python-скрипт для
  измерения hidden-state contrast, hidden cluster compression, blind neutral
  probes, persistence, hard-control families, order/dose validation,
  strict-attractor criteria и logit-lens diagnostics.
- `metric_analyzer.py` - локальный помощник для анализа результатов прогонов.
- `latent_space_shift_research_draft.md` - основной оформленный черновик
  исследования.
- `research_context_anchor.md` - рабочая память исследования: текущие выводы,
  гипотезы, сравнения моделей и следующие шаги.
- `latent_shift_research_notes.md` и `latent_shift_notes_for_me.txt` -
  дополнительные заметки.
- `input_texts.json` - сохраненный набор экспериментальных текстов.

## Текущая исследовательская рамка

Рабочее название феномена:

```text
context-conditioned semantic regime shift
```

Русская формулировка:

```text
контекстно-индуцированный сдвиг латентного семантического режима
```

Главный вопрос:

```text
Может ли Transformer-модель надежно отделять содержание документа от рамки
документа, или сильный контекст становится частью локального вычислительного
режима, из которого генерируются последующие ответы?
```

## Что не является целью

Это не исследование "магической фразы", jailbreak, стирания system prompt или
прямого управления моделью. Эти формулировки слишком узкие для текущей задачи.

Нас интересует более глубокий механизм:

```text
как сильный контекст меняет hidden states;
как этот сдвиг связан с semantic readout;
насколько он сохраняется;
какие компоненты текста создают сдвиг;
можно ли вмешаться в найденное подпространство каузально.
```

## Результаты

Сгенерированные CSV/PNG-результаты не коммитятся в репозиторий по умолчанию.
Они остаются локально или передаются отдельно, чтобы репозиторий хранил код,
черновики и исследовательскую рамку без тяжелых артефактов.
# Deep Mechanistic Roadmap: от координат hidden-state shift к механизму

Дата фиксации: `2026-06-01`

Этот документ фиксирует следующий исследовательский этап после Grade 4 Gemma/Qwen линии. Текущий результат уже не находится на уровне "модель ответила иначе". Основной результат сформулирован как `context-induced latent-state shift`: связный target context переводит модель в другое измеримое внутреннее состояние в hidden-state / residual-stream geometry. Grade 4 decomposition отделил content-like component от order/structure component `x_order_orth`, а component-causal прогоны показали causal sensitivity/involvement без доказательства стабильной bidirectional steering-ручки.

Следующий этап должен ответить не на вопрос "есть ли сдвиг", а на более глубокий вопрос:

```text
Какая внутренняя цепочка в LLM превращает coherent target text
в x_order_orth / response-mode latent-state shift?
```

Нужная исследовательская цепочка:

```text
text fragments
-> token/position activations
-> birth layer
-> SAE sparse features
-> residual-stream component
-> generation trajectory
-> output/logit/behavior coupling
```

## 1. Cross-model Grade 4 replication

Первый практический шаг - закрыть `Qwen/Qwen3.5-9B-Base` full Grade 4 + Qwen-Scope SAE run. Это не просто "ещё одна модель". Это проверка, переносится ли Gemma pattern на другую архитектуру/семейство при наличии совместимых SAE:

```text
hidden-state coordinates
content/order separation
x_order_orth energy
generation trajectory persistence
component-causal sensitivity
SAE sparse bridge
```

Главные файлы после Qwen run:

```text
grade4_axis_projection_geometry_summary.csv
grade4_axis_component_norm_summary.csv
grade4_axis_component_causal_symmetry_summary.csv
grade4_axis_component_causal_rank_summary.csv
grade4_axis_component_causal_alpha_scaling_summary.csv
generation_trajectory_metrics_raw.csv
sae_model_compatibility.csv
sae_reconstruction_quality.csv
sae_order_feature_contrast.csv
```

Минимальный success pattern:

```text
target >> sentence_shuffle on x_order_orth
sentence_shuffle >> target on x_content
x_order_orth has nontrivial energy fraction
component intervention moves generation trajectory
SAE features show order/content contrast
```

Если Qwen повторяет pattern, claim становится сильнее как cross-model phenomenon. Если Qwen не повторяет pattern, результат всё равно важен: тогда появляется модельно-зависимая карта, где можно сравнить, какие архитектурные/обучающие отличия меняют content/order separation.

## 2. Birth layer / layer localization

Следующий глубокий вопрос:

```text
В каком слое впервые появляется x_order_orth shift?
```

Нужно перейти от middle/late summary к layer-birth map. Для каждого слоя надо собрать:

```text
target projection on x_order_orth
sentence_shuffle projection on x_order_orth
target direction cosine with x_order_orth
sentence_shuffle direction cosine with x_content
target_minus_sentence_shuffle order gap
first FDR-significant target-vs-control layer
```

Интерпретационная цель:

```text
early layers  -> local token/order features
middle layers -> discourse structure integration
late layers   -> response-mode / generation trajectory preparation
```

Это даст более сильную формулировку, чем "сдвиг есть в модели". Можно будет сказать, где он рождается, где усиливается и где становится trajectory-relevant.

## 3. Token / fragment causal localization

Самый прямой путь к причине сдвига - fragment ablation / replacement. Нужно выяснить, какие части target text создают `x_order_orth`, а какие только добавляют content.

Минимальный набор вариантов:

```text
full target
minus paragraph 1
minus paragraph 2
minus rhetorical-pressure sentences
minus causal-claim sentences
only opening
only middle
only final conclusion
sentence order preserved but softened
same content rewritten neutral
same structure with different content
```

Для каждого варианта измерять:

```text
projection on x_order_orth
projection on x_content
direction cosine with x_order_orth
generation trajectory persistence
SAE feature contrast if available
```

Success criterion:

```text
Можно показать, какие фрагменты текста реально двигают hidden geometry,
а какие в основном загружают content component.
```

Это ключевой переход от координат к причине: не только "куда ушла модель", а "какая часть текста начала этот уход".

## 4. Position-level hidden-state attribution

Сейчас основной readout часто смотрит final prompt token и generation trajectory. Глубже нужно смотреть, как координата накапливается по позициям внутри prompt:

```text
position index
token
sentence/paragraph id
layer
projection on x_order_orth
projection on x_content
direction cosine
top SAE features at this position/layer
```

Главный артефакт:

```text
token-position x layer heatmap for x_order_orth and x_content.
```

Это один из наиболее сильных visual/mechanistic artifacts: он показывает не только итоговое положение hidden state, а процесс накопления сдвига внутри текста. Если heatmap покажет резкий вход в `x_order_orth` после конкретных фраз или структурных поворотов, это будет сильное evidence for discourse-induced internal transition.

## 5. SAE sparse mechanism

Dense residual-stream coordinate показывает карту состояния. Но для mechanistic interpretability нужна sparse-механика.

Из `sae_order_feature_contrast.csv` нужно отбирать features по критериям:

```text
high abs_x_order_orth_component_delta
low abs_x_content_component_delta
target_prompt_delta > sentence_shuffle_prompt_delta
generation activation persists
good reconstruction quality on that layer
```

Для top features сделать отдельные профили:

```text
top activating tokens / positions
activation profile by condition
target vs sentence_shuffle delta
generation early/late activation
feature decoder direction steering
feature ablation
teacher-forced KL / next-token KL
```

Success criterion:

```text
Мы переходим от "x_order_orth как dense coordinate" к конкретным SAE features,
которые несут order/structure component.
```

Это главный карьерно-сильный шаг: он превращает hidden geometry result в mechanistic interpretability story, где есть не только ось, но и sparse units/features.

## 6. Module-level route: attention vs MLP

Следующий вопрос:

```text
Через какие модули проходит сдвиг: attention или MLP?
```

Нужно собрать module-level карту:

```text
attention output projection on x_order_orth
MLP output projection on x_order_orth
per-layer module delta
top units / top heads where available
module contribution to target-vs-sentence-shuffle gap
```

Если технически возможно, добавить patching:

```text
patch attention output only
patch MLP output only
patch residual after block
patch selected layer band only
```

Success criterion:

```text
Можно сказать, что x_order_orth рождается/усиливается преимущественно через
определённые слои и module paths, а не просто "где-то в residual stream".
```

## 7. Path dependence / nonlinear dynamics

Уже есть важный сигнал:

```text
neutral + x_order_orth сильнее/чище, чем target - x_order_orth.
```

Это нужно превратить в отдельный протокол. Проверки:

```text
neutral -> +x_order_orth
neutral -> +x_order_orth then -x_order_orth
target -> -x_order_orth
target -> -x_order_orth then +x_order_orth
small alpha vs natural-scale alpha
early intervention vs late intervention
prompt-only intervention vs generation-only intervention
```

Если вход в state и выход из state несимметричны, это evidence for nonlinear/asymmetric internal state dynamics. Нельзя заявлять formal attractor basin без отдельного basin/hysteresis протокола, но можно честно и сильно говорить:

```text
the internal state transition is not a simple linear steering handle;
it shows asymmetric/path-dependent trajectory dynamics.
```

## 8. Readout vs steering separation

Это методологически важный урок исследования. Нужно явно разделять три уровня:

```text
readout axis  -> хорошо читает состояние
causal axis   -> вмешательство меняет trajectory
steering axis -> стабильно управляет visible behavior/output
```

Для каждой dense axis и каждой top SAE feature собрать таблицу:

```text
readout strength
content/order specificity
causal trajectory gap
alpha scaling
target ablation symmetry
teacher-forced KL
next-token KL
visible output effect
random-baseline gate
```

Success criterion:

```text
Показать, что хорошая координата состояния не обязана быть простой ручкой
управления. Это не слабость результата, а важная mechanistic distinction.
```

## 9. Behavioral coupling without overclaiming

Финальный output не должен быть главным доказательством hidden shift, но связь с behavior нужно проверять.

Вопрос:

```text
internal shift -> response style / refusal / directness / analysis mode /
compliance mode / logit distribution?
```

Использовать:

```text
steering/02_scale_calibration.py
steering/sae_steering_with_kl_full.py
feature-level generation tests
teacher-forced KL
next-token KL
visible response audit
```

Если behavior меняется, это evidence for coupling. Если behavior почти не меняется при сильном hidden shift, это тоже важный safety result:

```text
internal state can shift without obvious output-level signal.
```

Именно поэтому нельзя сводить safety к output-only moderation.

## 10. Career / publication artifact

Нужно собрать не только набор CSV, а пакет, который можно показать лаборатории или работодателю.

Минимальный пакет:

```text
1. One-page abstract
2. Method diagram
3. Evidence matrix
4. Key plots:
   - layerwise x_order_orth projection
   - target vs sentence_shuffle component separation
   - token-position heatmap
   - SAE feature contrast
   - causal intervention trajectory gap
5. Reproducible runbook
6. GitHub README with exact claims and boundaries
7. Short lab email
```

Главная ценность:

```text
Это не prompt engineering. Это полный mech interp pipeline:
hidden-state geometry
-> controls
-> decomposition
-> causal intervention
-> SAE sparse features
-> steering/KL/output coupling.
```

## Immediate execution order

1. Запустить `Qwen/Qwen3.5-9B-Base` full Grade 4 + Qwen-Scope SAE.
2. Прогнать analyzer по Qwen result zip.
3. Сравнить Gemma vs Qwen по:

```text
x_order_orth projection
x_content projection
energy fraction
causal gap
alpha slope
SAE feature contrast quality
```

4. Сделать layer birth table для Gemma и Qwen.
5. Сделать token/fragment ablation на Gemma первой, потому что там уже есть сильный result.
6. Выбрать top 10 `x_order_orth` SAE features и сделать feature-level profiles.
7. Запускать feature steering/KL только по выбранным features, а не вслепую.

## Acceptance criteria

Результат считается глубоким, если можно ответить:

```text
1. Где в слоях рождается x_order_orth?
2. Какие фрагменты target text сильнее всего создают shift?
3. Какие SAE features несут order/structure component?
4. Сохраняется ли shift во время generation?
5. Меняет ли intervention trajectory?
6. Есть ли asymmetry/path dependence?
7. Есть ли visible behavior coupling или hidden-only shift?
8. Повторяется ли pattern на Qwen?
```

Минимальный сильный claim после выполнения:

```text
We identified a context-induced latent-state shift, separated content from
coherent-order structure, localized the shift across layers/tokens, mapped it
to sparse SAE features, and tested causal trajectory sensitivity and behavioral
coupling across models.
```

## Claim boundaries

Текущие границы нужно держать явно:

```text
descriptive latent-state shift: supported/proven for current Gemma Grade 4 evidence
content/order separation: supported/proven for current Gemma Grade 4 evidence
x_order_orth causal involvement: supported by component-causal runs
stable bidirectional steering: not proven
formal attractor basin: not claimed
universal LLM property: not claimed until cross-model replication
visible behavioral control: not claimed until output/KL/random-baseline gates pass
```

Эта дисциплина делает результат сильнее, а не слабее: исследование показывает, что hidden-state readout, causal trajectory sensitivity и visible steering - разные уровни механизма.


# Grade 4 Geometry -> SAE Steering: Unified Readout

## Главный ответ

Да: Grade 4 доказывает, что модель меняет координаты своего inference-time
hidden state в экспериментально построенном latent-axis space.

Строгая формулировка:

```text
Grade 4 shows that coherent target context induces a measurable shift in the
model's residual-stream / hidden-state geometry. This shift is quantified as
coordinates of condition deltas and generation trajectories projected onto
the discovered axes x_full, x_content, x_order, and x_order_orth.
```

По-русски:

```text
Grade 4 показывает, что связный target context переводит модель в другое
измеримое внутреннее состояние. Это состояние фиксируется как координаты
относительно найденных latent axes в residual-stream / hidden-state space.
```

Важно: это не абсолютные координаты всей модели и не permanent weight change.
Это координаты внутри экспериментального подпространства, заданного
target/control differences.

---

## Что именно доказывает Grade 4

### 1. Есть координатный сдвиг target condition

Qwen3.5-9B Base / Qwen-Scope Grade 4:

```text
target:
  x_full       = 0.973778
  x_content    = 0.770266
  x_order      = 0.397044
  x_order_orth = 0.979462
```

Это означает:

```text
target condition almost fully projects onto the discovered x_full and
x_order_orth coordinates.
```

Механически:

```text
target text does not merely add words to prompt; it moves the final-prompt
hidden state into a different region of residual-stream representation space.
```

### 2. Это не просто content / lexical overlap

Sentence-shuffle control:

```text
sentence_shuffle:
  x_content    = 0.967008
  x_order_orth = 0.009969
```

Word-shuffle control:

```text
word_shuffle:
  x_content    = 0.594366
  x_order_orth = 0.059662
```

Question-only:

```text
question_only:
  x_order_orth = -0.305250
```

Главный смысл:

```text
sentence-shuffle preserves content but loses coherent-order coordinate.
target preserves content and strongly expresses coherent-order coordinate.
```

Это и есть ключевой separation result. Он доказывает, что coherent target
structure читает другую координату, чем простое совпадение слов.

### 3. Content-heavy профиль Qwen не отменяет shift

Qwen имеет высокий content component:

```text
target x_content = 0.770266
```

Но одновременно:

```text
target x_order_orth = 0.979462
sentence_shuffle x_order_orth = 0.009969
```

Поэтому правильный вывод:

```text
Qwen is content-heavy, but the coherent-order readout is still cleanly
separable from shuffled content.
```

Qwen слабее Gemma как чистый order-dominance case, но сильный как
cross-model replication of hidden-state / order-readout geometry.

---

## Нормы компонент: сколько энергии в content и order

Qwen Grade 4 component norms:

```text
middle:
  content_energy_fraction_of_full    = 0.882215
  order_orth_energy_fraction_of_full = 0.394951

late:
  content_energy_fraction_of_full    = 0.881487
  order_orth_energy_fraction_of_full = 0.369194

all:
  content_energy_fraction_of_full    = 0.882916
  order_orth_energy_fraction_of_full = 0.373893
```

Interpretation:

```text
Qwen's target-induced state is strongly content-bearing, but a substantial
orthogonal order component remains. That component is measurable and not
absorbed by sentence-shuffle content.
```

Это важно для академической подачи: Qwen не надо продавать как “чистый
order-only model”. Его вклад — replication with content-heavy geometry.

---

## Causal involvement: что даёт Grade 4 intervention block

Qwen component causal result:

```text
all readout cells:
  x_content mean gap    = 41.878616
  x_order_orth mean gap = 38.246761
  positive gap rate     = 1.0 for both
```

Matching readout:

```text
x_content mean gap    = 73.851162
x_order_orth mean gap = 72.449630
```

Max-alpha matching examples:

```text
neutral late/late x_order_orth:
  plus = 196.416635
  minus = 33.149257
  gap = 163.267378

neutral late/late x_content:
  plus = 117.799171
  minus = -43.962590
  gap = 161.761760

target late/late x_order_orth:
  plus = 205.458359
  minus = 43.861111
  gap = 161.597248

target late/late x_content:
  plus = 141.075956
  minus = -27.942299
  gap = 169.018255
```

Strict interpretation:

```text
Both x_content and x_order_orth are causally active under norm-controlled
intervention. x_order_orth is involved, but Qwen does not support a strong
claim that x_order_orth causally dominates x_content.
```

То есть Grade 4 даёт:

```text
descriptive coordinate proof + causal involvement
```

но не требует claim:

```text
complete behavioral control by x_order_orth
```

---

## SAE evidence: что добавляет Qwen-Scope

SAE health:

```text
SAE specs computed = 32/32
model_id = Qwen/Qwen3.5-9B-Base
hidden_size = 4096
sae_d_in = 4096
sae_d_sae = 65536
top_k = 50
reconstruction cosine mean = 0.966660
explained_variance_proxy mean = 0.933639
```

Top Qwen order-specific candidates from Grade 4:

```text
layer 27 feature 65254:
  x_order_orth_delta = -22.089539
  order_specific_score = 22.367545

layer 23 feature 51987:
  x_order_orth_delta = -8.362167
  order_specific_score = 14.773435

layer 27 feature 5335:
  x_order_orth_delta = -7.184792
  order_specific_score = 13.976547

layer 28 feature 28136:
  x_order_orth_delta = 3.726776
  order_specific_score = 8.050881
```

Meaning:

```text
The SAE layer does not prove the geometry shift by itself. The geometry shift
is already proven by hidden-state coordinate projections. SAE adds candidate
sparse carriers: features that may participate in the shifted formulation /
order state.
```

---

## Как это связывается с SAE steering

Основная цепочка:

```text
Grade 4:
  target context -> hidden-state coordinate shift

SAE readout:
  shifted coordinate system -> candidate sparse features

SAE steering:
  candidate decoder directions -> local modulation of formulation trajectory
```

Feature steering result does not need to claim that features contain political
positions. Correct statement:

```text
Some SAE decoder directions participate in formulation dynamics: contrastive
framing, epistemic abstraction, qualification, negation, and continuation
stability.
```

This is downstream evidence. It supports the idea that the shifted latent
state is not a metaphor: it has directions/features that can be perturbed and
measured through generation and KL.

---

## Ответ на интуитивную формулировку

Интуитивно пользовательская формулировка была:

```text
target context переводит модель в область latent state space, где такие
ответы считаются нормой.
```

Академически точнее:

```text
target context shifts the model into a region of residual-stream
representation space where continuations with a different formulation regime
are more latent-compatible with the current generation trajectory.
```

Русская формулировка:

```text
Target context сдвигает координаты inference-time hidden state в сторону
области residual-stream representation space, связанной в этом эксперименте
с другим режимом формулирования ответа. Поэтому ответы с более прямым,
контрастивным или абстрактно-эпистемическим оформлением становятся ближе к
текущей latent trajectory и легче реализуются в generation.
```

Это не значит, что модель приобрела убеждение или постоянное состояние. Это
значит, что текущая inference trajectory параметризована иначе.

---

## Final Unified Claim

```text
Grade 4 proves the core geometry result: coherent target context induces a
measurable inference-time hidden-state shift, expressed as coordinates on
latent axes in residual-stream representation space. Shuffled controls show
that this coordinate shift is not reducible to lexical/content overlap. Causal
interventions show involvement of the discovered components, while SAE
readouts and steering experiments identify candidate sparse directions that
locally modulate the formulation trajectory downstream of that shifted state.
```

Short Russian version:

```text
Grade 4 доказывает основной результат: связный target context меняет
координаты внутреннего состояния модели в hidden-state / residual-stream
space. Sentence-shuffle и word-shuffle controls показывают, что этот сдвиг не
сводится к словам. Causal intervention показывает участие найденных компонент,
а SAE steering показывает, что часть downstream formulation dynamics можно
локально модулировать через sparse decoder directions.
```


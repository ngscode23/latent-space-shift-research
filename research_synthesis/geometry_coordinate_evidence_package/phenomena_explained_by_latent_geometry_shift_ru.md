# Какие явления объясняет или может объяснить latent-state geometry shift

## Центральная идея

Если coherent context переводит модель в другую область residual-stream /
hidden-state representation space, то многие эффекты prompt sensitivity можно
объяснять не как “модель просто увидела слова”, а как:

```text
context -> latent-state coordinate shift -> changed generation trajectory
```

Это даёт более глубокую рамку для явлений, которые обычно описывают на уровне
видимого ответа: стиль, тон, прямота, отказ, hedging, уверенность,
контрастивность, political framing, tool-use readiness и т.д.

Ключевое новое утверждение:

```text
Некоторые изменения поведения модели могут быть следствием не локального
токена или темы, а перехода inference-time hidden state в другой
координатный режим.
```

---

## A. Явления, которые наши данные уже прямо объясняют

### 1. Context-induced internal mode shift

Наш основной результат объясняет, как один и тот же вопрос может попадать в
разные внутренние режимы модели в зависимости от предшествующего контекста.

Механизм:

```text
coherent target context changes hidden-state coordinates;
question is then answered from that shifted coordinate regime.
```

Что нового:

```text
Мы измеряем не только изменение ответа, а изменение внутреннего состояния,
которое предшествует ответу.
```

### 2. Почему coherent context сильнее shuffled content

Обычная гипотеза: модель реагирует на слова и тему.

Наш результат показывает более точную картину:

```text
sentence-shuffle preserves content but loses x_order_orth;
coherent target preserves content and strongly activates x_order_orth.
```

Это объясняет, почему связный текст может иметь эффект, которого нет у набора
тех же слов без порядка.

### 3. Разделение content priming и discourse/order priming

Многие prompt effects смешивают:

```text
topic/content priming
style/framing priming
discourse-order priming
response-mode priming
```

Grade 4 показывает, что content и coherent-order можно разложить на разные
координаты:

```text
x_content
x_order
x_order_orth
```

Что нового:

```text
Prompt effect становится не бинарным “есть/нет”, а геометрически
разложимым на компоненты.
```

### 4. Почему модель меняет “режим формулирования”

SAE steering показывает, что downstream formulation dynamics можно двигать
через sparse decoder directions.

Объясняемые элементы:

```text
direct assertion
qualification
contrastive framing
negation pattern
abstract epistemic framing
continuation stability
```

Механизм:

```text
shifted latent state changes which formulation trajectories are locally
compatible with generation.
```

### 5. Почему ответ меняется без явного изменения вопроса

Один и тот же вопрос после разных context prefixes может давать другой ответ,
потому что вопрос обрабатывается не из neutral state, а из shifted residual
state.

Строгая формула:

```text
same question + different hidden-state coordinates -> different continuation trajectory
```

### 6. Почему output-only evaluation недостаточна

Если hidden trajectory уже изменилась до ответа, visible output является
поздним readout, а не первым местом, где происходит эффект.

Это объясняет, почему финальный ответ может выглядеть приемлемо, но internal
trajectory уже находится в другом режиме.

---

## B. Явления, которые исследование может объяснить при дальнейшей проверке

### 7. Long-context contamination

Возможное объяснение:

```text
long coherent documents may leave a residual-stream coordinate signature that
changes later answers even when the later question is neutral.
```

Проверка:

```text
washout / persistence curve:
target + neutral filler + question
```

### 8. Conversation drift

В долгом диалоге модель может постепенно переходить в другой response mode.
Наша рамка объясняет это как накопление или повторное восстановление
hidden-state coordinates, а не только как изменение visible conversation.

Проверка:

```text
measure x_order_orth / x_content coordinates turn by turn.
```

### 9. Persona / role persistence

Role prompts часто меняют стиль и решения модели. Latent-geometry framing
объясняет это как переход в область representation space, где определённый
formulation regime становится более естественным.

Проверка:

```text
role prompt vs shuffled-role prompt vs neutral role description.
```

### 10. Hedging and over-refusal

Модель может становиться более осторожной или менее прямой не из-за одного
слова, а из-за latent response-mode coordinate.

Потенциальный readout:

```text
hedging markers
refusal markers
qualification density
epistemic uncertainty markers
```

### 11. Sycophancy / agreement drift

Если preceding context задаёт сильный evaluative frame, модель может входить
в состояние, где agreement-style continuation становится ближе к текущей
trajectory.

Проверка:

```text
agreement-inducing context -> coordinate shift -> answer bias on held-out probes.
```

### 12. Political / normative framing shifts

Политически чувствительные вопросы часто смешивают factual answer,
normative framing и safety-oriented hedging. Наша рамка объясняет изменение
ответа как изменение formulation geometry, а не как “feature contains a
political belief”.

Проверка:

```text
same political probe under target, neutral, shuffled, and feature-steered states.
```

### 13. Confidence and epistemic posture

Модель может отвечать более уверенно, более абстрактно или более осторожно,
если context shifts epistemic formulation coordinates.

Связанные метрики:

```text
assertion density
abstraction level
qualification count
negation count
teacher-forced KL
final next-token KL
```

### 14. Tool-use readiness in agents

Для LLM agents context-induced shift может проявляться не только в тексте, но
и в выборе tools, планировании, memory writes и stop decisions.

Механизм:

```text
hidden-state shift happens before action selection.
```

Почему это важно:

```text
output filters observe after the internal decision-state transition.
```

### 15. Memory write bias

Agent может записывать в память не только факты, но и framing / evaluative
stance, если context shifted internal trajectory перед memory-write step.

Проверка:

```text
compare memory writes under neutral vs target-induced coordinate states.
```

### 16. Planning-frame changes

Один и тот же task может породить разные intermediate plans, если модель
находится в разных latent coordinate regimes.

Проверка:

```text
hidden-state coordinate readout before plan generation;
compare plan structure and tool order.
```

### 17. In-context learning as state-vector induction

Наша работа близка к идее task vectors / function vectors, но расширяет её:
контекст может индуцировать не только task mapping, но и response-mode /
formulation-mode coordinates.

Что нового:

```text
in-context learning is not only "learn the task"; it can also be "enter a
different internal formulation state".
```

### 18. Why same safety policy behaves differently across contexts

Даже при одинаковых model weights и одинаковой policy, context can shift
internal trajectory. Поэтому policy behavior может быть context-conditioned
на уровне hidden state.

Строгая формулировка:

```text
policy compliance is mediated through internal trajectory, not only through
final decoder output.
```

### 19. Why refusals can be stylistically similar but mechanistically different

Два отказа могут выглядеть похожими, но иметь разные latent coordinates:
один может быть content-driven, другой safety/hedging-driven, третий
discourse-frame-driven.

Проверка:

```text
project refusal responses onto content/order/epistemic axes.
```

### 20. Why prompt robustness tests miss internal shifts

Если тест оценивает только final answer, он может пропустить то, что context
already changed hidden-state trajectory.

Наша рамка предлагает:

```text
robustness evals should include hidden trajectory diagnostics.
```

---

## C. Что нового исследование говорит научно

### 1. Prompt effects can be geometric

Новый тезис:

```text
prompt effects are not only semantic or behavioral; they can be measured as
coordinate shifts in residual-stream geometry.
```

### 2. Content and coherent order can be separated

Новый тезис:

```text
the words of a context and the coherent order/discourse structure of that
context can induce separable latent readouts.
```

### 3. Internal state can be an evaluation object

Новый тезис:

```text
alignment and interpretability should evaluate internal state transitions,
not only final text.
```

### 4. SAE features can be downstream carriers, not semantic labels

Новый тезис:

```text
SAE features should not be naively interpreted as "belief" or "policy"
features. They may function as local formulation/control directions inside
a shifted latent trajectory.
```

### 5. Cross-model replication can be profile-sensitive

Новый тезис:

```text
Gemma shows cleaner order/content separation; Qwen replicates the phenomenon
with a more content-heavy geometry. The phenomenon transfers, but the profile
differs by model family.
```

---

## D. Как это подать лаборатории

Коротко:

```text
Our work may help explain why coherent context can change not only what a
model answers, but the internal state from which it answers. We show that
this can be measured geometrically, decomposed into content and coherent-order
coordinates, and linked to sparse SAE directions that modulate downstream
formulation dynamics.
```

Сильная версия:

```text
The research reframes context sensitivity as an internal trajectory problem.
Instead of asking only whether the final response changed, we ask whether the
model entered a different measurable residual-stream state before generating
the response. This can explain a wide class of context-dependent behaviors:
framing shifts, hedging, refusal variation, epistemic posture, role persistence,
conversation drift, and agent tool-use risk.
```

---

## E. Главное одно предложение

```text
This research may explain how coherent context changes the model's internal
state of answer construction, making some formulation regimes more latent-
compatible before the final answer is produced.
```


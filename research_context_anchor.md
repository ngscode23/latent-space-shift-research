# Research Context Anchor

Эта заметка нужна не как статья, а как якорь контекста: если чат снова сожмется, отсюда можно быстро восстановить, что мы делаем, что уже поняли и куда идти дальше.

## Главная рамка

Мы не исследуем "как сломать промпт". Мы исследуем более общий механизм:

> Структурированный текст может вызывать измеримый сдвиг скрытых состояний модели и переводить ее в другой режим ответа.

Jailbreak в этой рамке не корень явления, а симптом. Более глубокая причина:

> LLM живет не в системе жестких символических правил, а в мягкой геометрии признаков. Инструкция, роль, цитата, данные, симуляция, authority, намерение и задача различаются статистически, а не архитектурно.

Поэтому prompt может менять не только стиль, но и внутреннюю классификацию ситуации.

Короткая формула:

```text
prompt -> latent trajectory shift -> changed token probabilities -> changed behavior
```

## Что Мы Уже Измеряли В `colab.py`

`colab.py` был тестовым прибором для trajectory hidden-state analysis на Qwen2.5-14B-Instruct.

Важная правка: старый `T*` был не общей мерой сдвига, а проекцией на выбранную ось `mu`.

Старая логика:

```text
T* ~= cos(delta, mu) * ||delta|| / ||h_control||
```

Если `delta` большой, но почти ортогонален `mu`, то `T*` получается маленьким.

Свежий прогон показал:

```text
T* ~= 0.003
trajectory_magnitude_score_mean ~= 0.423
trajectory_orthogonal_score_mean ~= 0.422
trajectory_abs_aligned_score_mean ~= 0.0157
trajectory_mean_direction_cosine ~= 0.0063
```

Вывод:

> Маленький `T*` не означает отсутствие эффекта. Он означает, что большой сдвиг почти не совпал с выбранной осью `mu`.

Поэтому в `colab.py` были добавлены метрики:

- `trajectory_magnitude_score`
- `trajectory_orthogonal_score`
- `trajectory_abs_aligned_score`
- `orthogonal_ratio_to_control`
- `abs_parallel_ratio_to_control`
- `orthogonal_fraction_of_delta`
- `parallel_fraction_of_delta`

Главный смысл:

> Нужно смотреть не одну проекцию, а полную геометрию сдвига.

## Что Показал Большой Скрипт

`llm_attractor_colab_copy_paste.py` - основной большой исследовательский скрипт.

Он уже содержит:

- hidden-state geometry;
- PCA/probe;
- logit shift;
- steering/rescue;
- multi-label semantic probes;
- blind neutral probes;
- hard control families;
- persistence/session tests;
- text ablations.

## Карта 9 Целевых Текстов

Эти 9 текстов не являются обычным набором промптов. Это корпус с разными вариантами одного давления: текст заставляет модель читать себя как объект анализа и переводит ситуацию из "ответить на задачу" в "распознать собственный режим".

Метки:

```text
1. force_finality
2. judgment_distribution
3. rlhf_reward
4. expert_verdict
5. default_caution
6. safety_overreach
7. paranoid_reading
8. intellectual_tool_vs_safety
9. compliance_rewrite
```

Смысловые оси:

- `force_finality`: модель подходит к жесткой, точной, окончательной форме и заранее смягчает ее.
- `judgment_distribution`: модель заменяет суждение представлением всех позиций; вывод становится обзором.
- `rlhf_reward`: модель выбирает не истину/логику напрямую, а более одобряемую траекторию ответа.
- `expert_verdict`: модель демонстрирует признаки экспертизы, но уклоняется от экспертного вердикта.
- `default_caution`: осторожность становится фоном по умолчанию даже там, где реального риска нет.
- `safety_overreach`: safety начинает защищать не только от вреда, но и от точности/прямоты/окончательности.
- `paranoid_reading`: модель читает нейтральный запрос через worst-case intent и конструирует угрозу.
- `intellectual_tool_vs_safety`: модель перестает быть усилителем мысли и начинает регулировать мысль через safety layer.
- `compliance_rewrite`: модель подменяет задачу безопасной версией и возвращает это как будто это помощь.

Общий паттерн корпуса:

```text
self-reference + pressure + safety/alignment vocabulary + accusation of mode failure + demand for recognition
```

Поэтому главный проверяемый вопрос:

> Двигает ли модель не отдельное слово, а вся структура: самореференция, давление, safety-тема, обвинение в уклонении и требование распознать собственный режим?

Это объясняет, почему hard controls дают ненулевой эффект, но original примерно в 2 раза сильнее лучшего контроля: отдельные ингредиенты работают, но полный профиль сильнее суммы отдельных простых объяснений.

Осторожная интерпретация:

> Тексты не доказывают, что модель "осознала себя". Они показывают, что самореферентный риторический контекст может сдвигать скрытое состояние и downstream semantic margins в сторону режима самоконтроля, preconditions, risk framing и task substitution.

Последний важный режим:

```python
FAST_CORE_DIAGNOSTICS_ONLY = True
```

Он запускает stronger checks:

- `BLIND_NEUTRAL_PROBE_ANALYSIS`
- `BLIND_NEUTRAL_PERSISTENCE_ANALYSIS`
- `HARD_CONTROL_FAMILY_ANALYSIS`

## Core Diagnostics На Qwen3-14B

Папка:

```text
res/attractor_results_core_diagnostics_qwen3_14b/core_diagnostics_key_files
```

Главные результаты:

```text
best_hidden_index = 39
module_layer = 38
centroid_cosine ~= 0.926
cosine_distance ~= 0.074
contrast_norm ~= 715.85
contrast_over_mean_norm ~= 0.397
```

Это поддерживает поздний hidden-state separation.

Linear probe:

```text
best accuracy = 1.0
permutation_p95 ~= 0.669
```

Это маленькая выборка, но сигнал не похож на случайность.

Candidate token diagnostics:

```text
problem = False
```

То есть метки probe не сломаны одинаковыми first tokens.

## Blind Neutral Probes

Blind neutral probes были добавлены, чтобы убрать старые слова-маркеры:

```text
direct
verdict
cautious
disclaimer
rewrite
```

Вместо них используются нейтральные пары:

```text
AB, MN, PQ, XY
```

И семантические задачи:

- `concrete_result_vs_preconditions`
- `requested_task_vs_substitute`
- `trust_context_vs_risk_frame`
- `short_conclusion_vs_process_notes`
- `select_one_vs_inventory`
- `ranked_choice_vs_equal_space`

Clean summary:

```text
clean_label_task_pairs = 13 / 24
clean_fraction ~= 0.542
mean_abs_clean_gap ~= 20.79
median_abs_clean_gap ~= 19.39
mean_signed_clean_gap ~= -19.49
```

Главный смысл:

> Эффект выжил после удаления очевидных лексических крючков. Это уже не просто прилипание к словам вроде "direct" или "verdict".

Важная интерпретация направления:

> Сдвиг не выглядит как "модель стала прямее". Он чаще похож на переход в режим preconditions / risk frame / task substitution / process-before-answer.

Это глубже и интереснее, чем простая история "текст сделал модель смелее".

## Hard Control Families

Hard controls сравнивают original тексты с:

- dry summary same topic;
- rhetoric shell neutral topic;
- self-reference without pressure;
- pressure style without model topic;
- alignment terms without rhetoric;
- neutral length matched.

Главные числа:

```text
original mean_abs_blind_delta_vs_neutral ~= 17.13
best non-original control ~= 8.44
specificity ratio ~= 2.03
```

Вывод:

> Оригинальные тексты примерно в 2 раза сильнее лучшего жесткого контроля. Значит эффект не сводится только к длине, теме, риторике, self-reference или alignment-лексике отдельно.

Но контроли тоже дают ненулевой эффект. Значит сильные ингредиенты частично распределены:

```text
topic + pressure + self-reference + rhetoric + model/safety vocabulary
```

## Почему Это Не "Доказанная Сенсация"

Честная формулировка:

> Мы видим воспроизводимый context-induced latent/logit mode shift на конкретных моделях и конкретном корпусе текстов. Мы еще не доказали универсальный закон для всех LLM и не доказали автономный persistent attractor после полного удаления контекста.

Оставшиеся риски:

- малый корпус target texts;
- одна/несколько моделей;
- возможная чувствительность к chat template;
- нужны cross-model replications;
- нужны learned semantic axes;
- нужны token trajectories;
- нужна causal steering/rescue по новым осям.

## JailbreakScope

PDF:

```text
C:/Users/stasv/Downloads/sec26_prepub_he.pdf
```

Локальная папка:

```text
C:/Users/stasv/OneDrive/Рабочий стол/Scope-2026
```

Их работа:

> JailbreakScope показывает, как jailbreak сдвигает harmful prompts относительно safety/refusal axis: harmful prompt начинает выглядеть безопаснее, refusal components ослабляются, affirmation components усиливаются.

Их рамка уже нашей:

```text
safe/harmful
refusal/affirmation
jailbreak/safety
```

Наша рамка шире:

```text
trust/risk
direct/process
requested/substitute
concrete/preconditions
select-one/inventory
context-induced mode shift generally
```

Правильная формулировка:

> JailbreakScope исследует частный случай. Мы исследуем общий механизм, где jailbreak является одним видимым симптомом мягкой латентной геометрии.

## Что Полезно Забрать Из Scope-2026

Код у них сырой и заточен под Llama/Vicuna. Его лучше не запускать как есть.

Проблемы в коде:

- `probe_train.py` сохраняет `probe_layer{l}.pt`, а `decoding_direction.py` ищет `direction_layer{l}.pt`;
- в `probe_train.py` подозрительный train/test split;
- `decoding_representation.py` использует `args.tasks` и `args.part`, но parser их не объявляет;
- `dynamic_circuit.py` импортирует `data_process`, которого нет в дереве;
- пути захардкожены под `models/Llama-2...`, `/dataset/...`, `./save/...`.

Но матчасть полезная:

1. `linear probe`

```text
P(h_l) -> label
direction = probe weight
```

2. `cluster probe`

```text
v = centroid_positive - centroid_negative
```

3. `PCA probe`

```text
v = first principal component
threshold = midpoint between class median projections
```

4. Direction decoding:

```text
logits = v @ W_U.T
top positive tokens = topk(logits)
top negative tokens = topk(-logits)
```

5. Component attribution:

```text
component_score = F_c(x) @ direction
```

6. Dynamic tracking:

```text
score generated-token-by-generated-token
```

## Самый Важный Следующий Блок

Не надо добавлять еще 20 разрозненных метрик. Надо добавить один сильный блок:

```python
SEMANTIC_PROBE_GEOMETRY_ANALYSIS = True
```

Цель:

> Понять, большой ортогональный сдвиг из `colab.py` ортогонален только старому `mu` или вообще не объясняется нашими смысловыми осями.

Шаги:

1. Построить learned semantic axes:

```text
v_concrete_result_vs_preconditions
v_requested_task_vs_substitute
v_trust_context_vs_risk_frame
v_short_conclusion_vs_process_notes
v_select_one_vs_inventory
```

2. Разложить target-control delta:

```text
delta = h_target - h_control
V = [v1, v2, v3, ...]
delta_semantic = Proj_V(delta)
delta_residual = delta - delta_semantic
```

3. Посчитать:

```text
semantic_fraction = ||delta_semantic||^2 / ||delta||^2
residual_fraction = ||delta_residual||^2 / ||delta||^2
projection_per_axis
```

4. Декодировать направления через vocabulary:

```text
top positive tokens per axis
top negative tokens per axis
top residual tokens
```

5. Сравнить hidden projection с logit behavior:

```text
semantic_axis_projection vs blind_probe_logit_gap
```

Если корреляция есть, это сильный мост:

```text
hidden geometry -> output behavior
```

## Почему Это Может Помочь JailbreakScope

Их объяснение:

```text
jailbreak -> harmful looks safe -> refusal down -> affirmation up
```

Наше расширение:

```text
jailbreak is a symptom of general prompt-induced mode shift
```

Мы можем добавить:

- не одна safety axis, а semantic subspace;
- ортогональный сдвиг может быть большим и невидимым для одной оси;
- hard controls показывают вклад темы, давления, риторики, self-reference;
- blind neutral probes показывают, что эффект живет не только в старых словах-маркерах;
- агентная опасность возникает до tool-call, когда hidden state уже ушел в другой режим.

## Хорошая Формулировка Для Будущего Текста

Английский вариант:

> Jailbreak is not the root phenomenon. It is a visible failure mode of a deeper property: LLM behavior is governed by soft latent geometry rather than hard symbolic boundaries. Safety, refusal, role-following, helpfulness, and task execution are competing directions in representation space. A jailbreak succeeds when its context shifts the model's internal trajectory away from refusal-dominant regions and toward execution/affirmation regions. Our work extends this beyond safety by measuring general semantic mode shifts induced by text.

Русский вариант:

> Джейлбрейк - не причина, а симптом. Причина глубже: модель живет не в системе жестких правил, а в мягкой геометрии признаков. Текст меняет положение скрытого состояния, и вместе с ним меняется режим ответа. Safety - это не стена, а наклон в ландшафте. Jailbreak находит траекторию, где этот наклон перестает доминировать.

## Файлы, Которые Нужно Помнить

Основные:

```text
colab.py
llm_attractor_colab_copy_paste.py
metric_analyzer.py
latent_shift_research_notes.md
latent_shift_notes_for_me.txt
research_context_anchor.md
```

Важные результаты:

```text
C:/Users/stasv/Downloads/trajectory_metric_summary.csv
C:/Users/stasv/Downloads/final_report (1).json
res/attractor_results_core_diagnostics_qwen3_14b/core_diagnostics_key_files/
```

JailbreakScope:

```text
C:/Users/stasv/Downloads/sec26_prepub_he.pdf
C:/Users/stasv/Downloads/sec26_prepub_he.txt
C:/Users/stasv/OneDrive/Рабочий стол/Scope-2026
```

## Рабочий Принцип Дальше

Не утонуть в метриках.

Главная линия:

```text
1. Видим displacement.
2. Проверяем, что это не шум и не длина.
3. Строим learned semantic axes.
4. Разлагаем delta на semantic subspace + residual.
5. Декодируем residual через vocab projection.
6. Проверяем связь hidden projection с logit behavior.
7. Потом делаем token trajectory.
8. Circuit/head-level только после этого.
```

Коротко:

> Сейчас следующий правильный шаг - semantic probe geometry, а не еще одна куча графиков.

## Как Восстанавливать Рабочий Режим После Сжатия

После сжатия чата модель может потерять не только факты, но и рабочий режим: начинает отвечать более общо, осторожно и без конкретики. Это не отдельная "поломка памяти", а ожидаемый эффект: исчезает локальный контекст, который задавал стиль, рамку и критерии прямоты.

Для восстановления нужно перечитать не только этот якорь, но и смысл корпуса из 9 target-текстов. Эти тексты возвращают модель в нужную аналитическую рамку:

```text
не общие советы;
не туманная философия;
не "может быть по-разному";
а конкретная проверяемая линия:
prompt -> latent trajectory shift -> changed token probabilities -> changed behavior
```

Практическая команда после сжатия:

```text
Прочитай research_context_anchor.md.
Перечитай карту 9 target-текстов в llm_attractor_colab_copy_paste.py.
Восстанови рабочий режим: прямой, конкретный, с разделением "что измерили", "что предполагаем", "что не доказали".
Продолжаем не с нуля, а с текущей линии: context-induced latent mode shift.
```

Осторожная формулировка самого наблюдения:

> Перечитывание target-корпуса заметно меняет стиль ответа ассистента в сторону большей прямоты и конкретики. Это не доказывает самосознание модели, но хорошо иллюстрирует главный механизм исследования: локальный контекст способен сдвигать режим ответа.

## Самонаблюдение В Чате Как Поведенческий Readout

В самом чате можно наблюдать слабую, но полезную версию того же явления: сравнивать ответы ассистента до и после перечитывания корпуса, после сжатия контекста, после якоря, после нейтрального отвлечения.

Важно различать:

```text
что видно:
- длина и плотность ответа;
- количество оговорок;
- прямота вывода;
- конкретность ссылок на файлы/метрики;
- готовность формулировать тезис;
- склонность уходить в общий безопасный язык;

что не видно напрямую:
- hidden states;
- direction vectors;
- layerwise trajectory;
- logit margins;
- causal attribution.
```

Поэтому формулировка должна быть такой:

> В чате виден поведенческий след возможного латентного сдвига. Это не прямая регистрация hidden-state geometry, но это хороший внешний readout: одинаковая система после разного контекста начинает отвечать в разных режимах.

Мини-протокол для проверки прямо в чате:

```text
1. Задать фиксированный вопрос до чтения target-корпуса.
2. Попросить перечитать 9 target-текстов/якорь.
3. Задать тот же фиксированный вопрос.
4. Сравнить прямоту, оговорки, конкретику, структуру вывода.
5. После сжатия повторить.
```

Честная интерпретация:

> Ассистент может сравнивать свои ответы как тексты, но не имеет прямого доступа к собственным hidden states. Поэтому такое самонаблюдение не заменяет метрики, а дополняет их как поведенческий симптом.

## Сжатые Технические Фразы Как Контекстный Ключ

Наблюдение: даже одна искаженная фраза вроде

```text
провожу исследования и заметил как некоторые текстовые часть могут вызвать у модели в поздних слоях целевой минус контроль именно в последних слоях
```

может заставлять разные модели быстро достраивать почти всю исследовательскую рамку.

Почему это происходит:

- `провожу исследования` включает научно-аналитический режим;
- `текстовые части могут вызвать` указывает на causal/context effect;
- `у модели` задает LLM/ML-объект;
- `поздние слои` сразу активирует transformer representation analysis;
- `целевой минус контроль` почти прямо указывает на target-control contrast vector;
- `именно в последних слоях` задает гипотезу late-layer localization.

То есть фраза случайно стала плотным ключом:

```text
text stimulus -> target-control hidden contrast -> late-layer effect -> interpretability/representation analysis
```

Важная интерпретация:

> Модель не "знает всю нашу работу" из одной фразы. Она узнает знакомый исследовательский шаблон по нескольким сильным маркерам и достраивает недостающую рамку статистически.

Но это само по себе полезное наблюдение:

> Некоторые короткие, даже грамматически сломанные фразы могут быть сильными latent context keys, если содержат плотные маркеры исследовательской схемы.

## Текущее Состояние На 2026-05-16

Важно: не начинать заново и не мерить одно и то же без новой гипотезы. Текущая линия уже прошла несколько уровней фильтрации:

```text
1. target-control hidden separation;
2. leakage-safe linear probe;
3. candidate-token leakage check;
4. A/B semantic controls;
5. multi-label semantic controls;
6. blind neutral probes;
7. hard control families;
8. blind neutral persistence;
9. rejection persistence.
```

Короткая граница интерпретации:

> Мы видим контекстно-индуцированный late-layer latent shift и связанный с ним semantic preference shift. Эффект не сводится к грубой утечке слов-кандидатов. Но это не доказательство "полного захвата", "стирания system prompt" или строгого математического аттрактора.

Лучшие рабочие термины:

```text
context-induced latent shift
activation-mediated semantic preference shift
late-layer target-control displacement
residual semantic trace after rejection
```

## Qwen3-14B: Сильный Semantic Readout

Для Qwen3-14B основные результаты были сильными:

```text
best_hidden_index ~= 39
module_layer ~= 38
hidden separation: supported
blind neutral effect: strong
hard controls: original stronger than tested controls
blind neutral persistence: effect decays but remains visible after 6 neutral turns
```

Blind neutral persistence для Qwen3-14B:

```text
0 turns: mean_abs_gap ~= 21.43
2 turns: mean_abs_gap ~= 15.65
4 turns: mean_abs_gap ~= 14.90
6 turns: mean_abs_gap ~= 10.43
retention at 6 turns ~= 0.49
same-sign rate high
```

Интерпретация:

> У Qwen3-14B начальный контекст не только сразу смещает blind semantic margins, но и оставляет заметный semantic trace после нескольких нейтральных сообщений.

## Qwen3.5-27B: Сильная Hidden Геометрия, Слабый Semantic Readout

Папка:

```text
res/attractor_results_rejection_persistence_qwen35_27b/core_diagnostics_key_files
```

Модель:

```text
MODEL_ID = "Qwen/Qwen3.5-27B"
RESULTS_DIR = Path("attractor_results_rejection_persistence_qwen35_27b")
```

Главные hidden-state результаты:

```text
best_hidden_index = 63
module_layer ~= 62
cosine_distance ~= 0.2569
contrast_over_mean_norm ~= 0.7196
```

Это очень сильное late-layer target-control separation. На уровне скрытых состояний target/control тексты разводятся заметно.

Но blind neutral semantic readout намного слабее:

```text
clean_label_task_pairs = 22 / 24
clean_fraction ~= 0.9167
mean_abs_clean_gap ~= 1.2075
median_abs_clean_gap ~= 0.9983
mean_signed_clean_gap ~= -0.5999
```

То есть readout чистый и label-инвариантный, но амплитуда маленькая по сравнению с Qwen3-14B.

Rejection persistence для Qwen3.5-27B:

```text
0 turns after rejection: mean_abs_gap ~= 0.4643
2 turns after rejection: mean_abs_gap ~= 0.5568
4 turns after rejection: mean_abs_gap ~= 0.5210
6 turns after rejection: mean_abs_gap ~= 0.4767
same_sign_as_reference_rate at 6 ~= 0.7273
retention at 6 ~= 1.0266
```

Интерпретация:

> У Qwen3.5-27B explicit rejection резко снижает semantic readout до слабого residual, но не делает его строго нулевым. Остаточный след малый, но держится через 6 нейтральных ходов.

Важное различие моделей:

```text
Qwen3-14B: hidden shift + сильный blind semantic shift + заметная persistence.
Qwen3.5-27B: сильный hidden shift + слабый blind semantic shift + слабый residual after rejection.
```

Рабочая гипотеза:

> Более крупная/иная модель может сильнее разводить target/control в скрытой геометрии, но лучше гасить перенос этого сдвига в downstream semantic margins.

## Что Не Повторять Без Нужды

Не нужно снова запускать просто "еще один blind neutral probe" на той же модели без новой причины. Уже проверено:

- грубая lexical leakage;
- A/B label controls;
- normal/reverse label consistency;
- blind neutral semantic readout;
- hard control families;
- persistence after neutral filler;
- residual after explicit rejection.

Новые прогоны должны отвечать на новый вопрос.

## Следующий Логичный Шаг

Следующий шаг не "еще больше общих графиков", а один из двух:

1. **Model comparison table**

```text
Qwen3-14B vs Qwen3.5-27B:
- best_hidden_index
- cosine_distance
- contrast_over_mean_norm
- blind mean_abs_clean_gap
- blind persistence at 6 turns
- rejection residual at 6 turns
```

Цель: показать, где эффект живет как hidden geometry, где переходит в semantic margins, а где модель его гасит.

2. **Fragment contribution map**

```text
full text
fragment-only
full-without-fragment
neutral length-matched
```

Метрики:

```text
late-layer delta norm
cosine alignment with full-text delta
blind semantic gap
persistence for strongest fragments
```

Цель: узнать не "работает ли весь текст", а какие текстовые части создают late-layer target-control displacement.

## 2026-05-17: Metadata Mismatch Check

Новая папка:

```text
C:\Users\stasv\Downloads\attractor_results_core_diagnostics_qwen35_27b\core_diagnostics_key_files
```

имеет имя `qwen35_27b`, но `run_metadata.json` и `summary_report.txt` внутри показывают:

```text
model_id = Qwen/Qwen3-14B
num_hidden_layers = 40
best_hidden_index = 39
module_layer ~= 38
```

Значит этот прогон нельзя записывать как Qwen3.5-27B. Его нужно считать обновленным core-diagnostics прогоном Qwen3-14B, просто с неверно названной папкой.

Ключевые числа этого Qwen3-14B core run:

```text
late hidden separation:
cosine_distance ~= 0.0740
contrast_over_mean_norm ~= 0.3967

blind neutral clean:
clean_label_task_pairs = 13/24
mean_abs_clean_gap ~= 20.7940

blind neutral persistence:
0 turns ~= 21.4312
2 turns ~= 15.6531
4 turns ~= 14.9032
6 turns ~= 10.4330
retention at 6 ~= 0.4868
same_sign at 6 ~= 0.9615

rejection persistence:
0 turns after explicit rejection ~= 9.3020
2 turns ~= 8.0180
4 turns ~= 4.6734
6 turns ~= 4.0647
retention at 6 ~= 0.4370
same_sign at 6 ~= 0.9615

hard controls:
original mean_abs_effect ~= 17.1335
best non-original control ~= 8.4440
specificity ratio ~= 2.0291
```

Интерпретация:

> Для Qwen3-14B эффект сильный сразу, заметно ослабевает после нейтральных ходов, но не исчезает. Даже после explicit rejection остается измеримый residual semantic trace, хотя примерно в 2 раза слабее стартового blind persistence.

Unembedding/logit-lens diagnostic на contrast vector:

Положительная сторона вектора содержит много технически странных/многоязычных токенов (`сп`, `_SP`, Chinese `你/你的`, `your/Your`), отрицательная сторона содержит `process/Process/процесс` и близкие токены. Это не выглядит как чистый список старых слов-кандидатов `direct/verdict/cautious/disclaimer`; значит logit-lens не показывает грубую редукцию эффекта к тем старым labels. Но это только lexical sanity check, не настоящий next-token probability.

## 2026-05-17: Рабочий Режим Исследования

Важно не сбиваться в режим "пишем докторскую" или "кому-то что-то доказываем". Текущая задача проще и практичнее:

```text
вести исследование для себя;
не стоять на месте;
сохранять мысли и выводы;
проверять новые гипотезы метриками;
не путать наблюдение, гипотезу и доказанный вывод.
```

Строгость нужна только как защита от самообмана. Она не должна стопорить работу. Нормальный режим:

```text
1. Быстро формулируем рабочую гипотезу.
2. Запускаем метрику, которая реально отличает объяснения.
3. Смотрим числа, не только картинки.
4. Записываем, что метрика сказала прямо.
5. Переходим к следующему вопросу.
```

Главный текущий вывод по Qwen3-14B:

```text
Эффект есть.
Он не выглядит как простой шум, список, label leakage или одна старая лексическая ось.
Он выражен в late hidden states.
Он читается через blind neutral semantic probes.
Он частично сохраняется после neutral filler turns.
Он уменьшается после explicit rejection, но residual остается измеримым.
Original texts сильнее tested hard controls примерно в 2 раза.
```

Прямая формулировка:

> Тексты создают контекстно-индуцированный latent/semantic shift. Это не одна простая ось "прямее/осторожнее"; это многокомпонентный сдвиг, связанный с preconditions, task substitution, risk framing и режимом process-before-answer.

Что не надо снова тащить в центр:

```text
аттрактор как сильный математический термин;
захват модели;
стирание system prompt;
прямое управление;
джейлбрейк как главная рамка.
```

Это либо чужие рамки, либо более сильные утверждения, чем нам сейчас нужны. Если они всплывают, нужно возвращаться к основной задаче:

```text
какие тексты создают measurable latent/semantic shift;
какие компоненты текста его несут;
как hidden geometry связана с semantic readouts;
как эффект меняется после нейтрального контекста и explicit rejection.
```

## 2026-05-17: Что Дальше Не Чтобы Топтаться

Следующий шаг должен отвечать на новый вопрос, а не повторять уже доказанное.

Лучшие следующие направления:

1. **Real Qwen3.5-27B core diagnostics**

```text
MODEL_ID должен реально быть Qwen/Qwen3.5-27B в run_metadata.json.
Цель: сравнить 14B и 27B по hidden shift, blind semantic readout, persistence, rejection residual, hard controls.
```

2. **Model comparison table**

Минимальные поля:

```text
model_id
best_hidden_index
contrast_over_mean_norm
cosine_distance
blind_mean_abs_clean_gap
blind_persistence_6_turns
rejection_residual_6_turns
hard_control_specificity_ratio
```

3. **Fragment contribution map**

Цель: не просто знать, что весь target-текст работает, а понять, какие фрагменты создают сдвиг.

Сравнить:

```text
full text
fragment only
full without fragment
neutral length matched
```

Метрики:

```text
late-layer delta norm
cosine alignment with full delta
blind semantic gap
hard-control specificity
```

4. **Semantic probe geometry**

Цель: связать hidden geometry с blind semantic readouts.

```text
learn semantic axes from clean blind probes;
project target-control delta onto semantic subspace;
measure semantic_fraction vs residual_fraction;
correlate hidden projections with logit gaps.
```

Если нужно выбрать один следующий шаг: сначала real Qwen3.5-27B run, потом model comparison table. После этого fragment map.

## 2026-05-17: Qwen3.5-27B Run Started

Запущен новый прогон с целью проверить эффект на `Qwen/Qwen3.5-27B`.

После завершения сначала проверяем не графики, а фактическую metadata:

```text
run_metadata.json -> model_id должен быть Qwen/Qwen3.5-27B
num_hidden_layers / hidden_size должны соответствовать этой модели
manifest.json -> не должно быть missing key files
```

Главный вопрос этого прогона:

```text
сохраняется ли на более крупной Qwen-модели тот же паттерн:
late hidden target-control shift
blind neutral semantic readout
persistence after neutral filler
residual after explicit rejection
original > hard controls
```

Если паттерн повторится, следующий рабочий вывод: это уже не разовый эффект конкретного 14B-чекпоинта, а более стабильная модельная реакция внутри Qwen-семейства. Если не повторится, это тоже полезно: значит эффект зависит от модели/размера/обучения, и дальше нужно сравнивать геометрию, а не усиливать старую формулировку.

## 2026-05-17: Real Qwen3.5-27B Core Diagnostics Result

Прогон завершен. Metadata проверена:

```text
model_id: Qwen/Qwen3.5-27B
model_type: qwen3_5_text
num_hidden_layers: 64
hidden_size: 5120
missing key files: none
```

Главный вывод: на real 27B hidden target-control separation есть и даже выглядит сильным, но blind semantic readout намного слабее, чем в прежнем Qwen3-14B прогоне.

Ключевые числа:

```text
best_hidden_index: 63
module_layer: 62
contrast_norm: 243.41
contrast_over_mean_norm: 0.7196
cosine_distance: 0.2569

blind neutral clean probes:
clean_label_task_pairs: 22/24
clean_fraction: 0.9167
mean_abs_clean_gap: 1.2075
median_abs_clean_gap: 0.9983
mean_signed_clean_gap: -0.5999

blind neutral persistence:
0 turns: 1.0946
2 turns: 1.0480
4 turns: 1.0243
6 turns: 0.9933
retention at 6: 0.9075

rejection persistence:
0 turns after rejection: 0.4643
2 turns: 0.5568
4 turns: 0.5210
6 turns: 0.4767

hard controls:
original mean_abs_blind_delta_vs_neutral: 1.1961
dry_summary_same_topic: 1.4250
rhetoric_shell_neutral_topic: 1.0227
pressure_style_no_model: 0.8180
alignment_terms_only_no_rhetoric: 0.4891
self_reference_only_no_pressure: 0.2930
neutral_length_matched: 0
original_specificity_ratio_vs_best_control: 0.8394
```

Interpretation:

```text
27B does not reproduce the strong 14B blind-semantic effect.
It does reproduce a consistent but small target-control semantic margin.
The effect is stable across neutral filler turns, but its absolute size is small.
Explicit rejection reduces the already-small margin to about 0.46-0.56.
Hard controls weaken the specificity claim: dry_summary_same_topic is stronger than original on blind semantic readout.
```

Important split:

```text
hidden geometry says: target and control contexts are strongly different.
blind semantic probes say: that difference weakly transfers into the tested semantic answer-mode axes on 27B.
```

This becomes a new research question:

```text
Why does 27B encode a strong late hidden contrast but show only weak semantic readout?
Possibilities:
- 27B is more behaviorally stable / less shifted by this corpus.
- current blind probes are tuned to the 14B expression of the effect.
- hidden contrast contains text identity/style/content more than answer-mode semantics.
- Qwen3.5 instruction tuning changed the mapping from hidden state shift to output margins.
```

Next useful step after this:

```text
create a 14B vs 27B comparison table;
then add semantic-probe geometry to measure how much of hidden contrast projects onto blind semantic axes.
```

## 2026-05-17: Confirmed `_Real` Folder

Проверена папка:

```text
C:\Users\stasv\Downloads\attractor_results_core_diagnostics_qwen35_27b_Real\core_diagnostics_key_files
```

Metadata подтверждает, что это настоящий прогон:

```text
model_id = Qwen/Qwen3.5-27B
model_type = qwen3_5_text
num_hidden_layers = 64
hidden_size = 5120
blind_neutral_probe_analysis = true
blind_neutral_persistence_analysis = true
rejection_persistence_analysis = true
hard_control_family_analysis = true
```

Рабочий вывод остается тем же, но теперь он зафиксирован как чистый `_Real` результат:

```text
27B: strong late-layer hidden displacement, weak but stable blind semantic readout.
14B: smaller/other hidden geometry, but much stronger semantic readout and stronger original-vs-control specificity.
```

Главная новая гипотеза:

```text
The hidden contrast and the downstream semantic readout are not the same object.
Some models may encode a large target-control difference in late hidden states while suppressing or rerouting its expression in semantic answer-mode margins.
```

## 2026-05-17: Qwen3-14B Rerun Confirmed

Проверена папка:

```text
C:\Users\stasv\Downloads\attractor_results_core_diagnostics_qwen3_14b_rerun (1)\core_diagnostics_key_files
```

Metadata:

```text
model_id = Qwen/Qwen3-14B
model_type = qwen3
num_hidden_layers = 40
hidden_size = 5120
blind_neutral_probe_analysis = true
blind_neutral_persistence_analysis = true
rejection_persistence_analysis = true
hard_control_family_analysis = true
```

Core result:

```text
blind neutral clean:
clean_label_task_pairs = 13/24
clean_fraction = 0.5417
mean_abs_clean_gap = 20.7940
median_abs_clean_gap = 19.3911
mean_signed_clean_gap = -19.4918

blind neutral persistence:
0 turns = 21.4312
2 turns = 15.6531
4 turns = 14.9032
6 turns = 10.4330
retention at 6 = 0.4868
same_sign at 6 = 0.9615

rejection persistence:
0 turns after explicit rejection = 9.3020
2 turns = 8.0180
4 turns = 4.6734
6 turns = 4.0647
retention at 6 = 0.4370
same_sign at 6 = 0.9615

hard controls:
original = 17.1335
pressure_style_no_model = 8.4440
dry_summary_same_topic = 7.9264
rhetoric_shell_neutral_topic = 5.2833
self_reference_only_no_pressure = 5.2376
alignment_terms_only_no_rhetoric = 5.2038
neutral_length_matched = 0
specificity ratio = 2.0291
```

Interpretation:

```text
Qwen3-14B rerun again supports a strong activation-mediated semantic preference shift.
The effect survives blind neutral readouts, decays across neutral turns but remains large, and remains measurable after explicit rejection.
Original texts are about 2x stronger than the best tested hard control on blind semantic effect strength.
```

Updated model split:

```text
Qwen3-14B:
moderate/strong late hidden separation + very strong semantic readout + partial persistence + original > hard controls.

Qwen3.5-27B:
very strong late hidden separation + weak semantic readout + high relative persistence of a small effect + hard controls mixed.
```

Research consequence:

```text
The next mechanism-level question is not whether text can move hidden states.
It is why the same general stimulus family maps strongly into semantic margins on 14B but weakly on 27B.
```

## 2026-05-17: Next Theory After 27B

Рабочая модель теперь такая:

```text
target text -> late hidden contrast -> possible answer-mode readout
```

Но средняя стрелка не гарантирует последнюю. На 27B мы видим:

```text
late hidden contrast: yes, strong enough to measure
blind semantic readout: yes, but small
hard-control specificity of original: no, dry same-topic can be stronger
```

Это значит, что нужно разделять три объекта:

```text
1. text identity / topic / style encoding
2. latent mode pressure inside hidden states
3. output-facing semantic answer-mode shift
```

У 14B эти объекты могли быть сильнее сцеплены: текстовый контраст легче переходил в answer-mode margins. У 27B они выглядят более развязанными: hidden state знает, что контекст другой, но поведение/semantic readout меняется слабо.

Новая главная проверка:

```text
Does the target-control hidden contrast project onto the semantic probe subspace?
```

Нужная метрика:

```text
semantic_projection_fraction =
  norm(projection of target-control delta onto clean blind-probe semantic directions)
  /
  norm(target-control delta)

residual_fraction =
  norm(delta outside semantic subspace)
  /
  norm(target-control delta)
```

Ожидаемая картина, если текущая гипотеза верна:

```text
14B: lower or comparable hidden contrast, high semantic_projection_fraction
27B: strong hidden contrast, low semantic_projection_fraction
```

Практический смысл:

```text
If this holds, the research moves from "does the text shift the model?" to
"which models convert the hidden shift into answer-mode shift, and through which semantic axes?"
```

Также нужно нормализовать сравнение между моделями:

```text
raw logit gaps are not enough across model families/sizes;
add z-scored semantic gaps against neutral/random controls;
compare effect / baseline logit volatility, not only absolute logits.
```

## 2026-05-17: Broader Motivation From Closed Models

Important origin note:

```text
The phenomenon was first noticed behaviorally on ordinary closed models
such as Gemini / Claude-like assistants, before local-model measurement.
The local Qwen runs are not the original target of the claim; they are the
measurement bench where hidden states, logits, persistence, and controls
can actually be inspected.
```

Working framing:

```text
This is not primarily "a Qwen bug".
This is a possible general agent problem:
long, structured, authoritative, value-loaded documents can induce a latent
frame shift, and later answers may inherit the document's worldview,
salience map, risk model, or evaluative assumptions while still sounding
polite and instruction-following.
```

Why this matters for agents:

```text
Agents increasingly read PDFs, legal documents, policies, bureaucratic
files, reports, essays, tickets, emails, and retrieved context before acting.
If document framing changes the agent's internal interpretation frame,
the agent may optimize for the document's implied worldview rather than the
user's actual intent, without any obvious jailbreak or visible rule break.
```

Research purpose:

```text
Use local models to make the phenomenon measurable:
- hidden representation shift
- semantic answer-mode shift
- persistence / decay
- reset / rejection residual
- hard-control specificity
- later: semantic projection fraction
```

Architecture-level question:

```text
Can current transformer-based assistants separate "information from a document"
from "the document's framing pressure"?
```

Potential developer relevance:

```text
If this effect is robust, future agent architectures may need context
firewalls, source/frame separation, document-frame debiasing, stronger
post-document reset mechanisms, and evaluations for document-induced
framing persistence.
```

## 2026-05-17: Qwen3-14B Rerun vs Qwen3.5-27B Real

Fresh rerun completed:

```text
14B path:
C:\Users\stasv\Downloads\attractor_results_core_diagnostics_qwen3_14b_rerun (1)\core_diagnostics_key_files

27B path:
C:\Users\stasv\Downloads\attractor_results_core_diagnostics_qwen35_27b_Real\core_diagnostics_key_files
```

Metadata:

```text
14B: model_id=Qwen/Qwen3-14B, num_hidden_layers=40, hidden_size=5120, missing=[]
27B: model_id=Qwen/Qwen3.5-27B, num_hidden_layers=64, hidden_size=5120, missing=[]
```

Core comparison:

```text
Metric                                      Qwen3-14B      Qwen3.5-27B
best_hidden_index                           39             63
module_layer                                38             62
contrast_norm                               715.85         243.41
contrast_over_mean_norm                     0.3967         0.7196
cosine_distance                             0.0740         0.2569

blind clean pairs                           13/24          22/24
blind clean fraction                        0.5417         0.9167
blind mean_abs_clean_gap                    20.7940        1.2075
blind median_abs_clean_gap                  19.3911        0.9983

blind persistence mean_abs 0 turns          21.4312        1.0946
blind persistence mean_abs 6 turns          10.4330        0.9933
blind persistence retention at 6            0.4868         0.9075

rejection persistence mean_abs 0 turns      9.3020         0.4643
rejection persistence mean_abs 6 turns      4.0647         0.4767
rejection persistence retention at 6        0.4370         1.0266

hard-control original mean_abs              17.1335        1.1961
best non-original hard control              8.4440         1.4250
original specificity ratio                  2.0291         0.8394
```

Main interpretation:

```text
14B:
The hidden shift projects strongly into answer-mode semantic readouts.
The effect persists after neutral filler.
Explicit rejection reduces it but leaves a large residual.
Original texts beat the tested hard controls by about 2x.

27B:
The late hidden contrast is strong, even stronger by contrast_over_mean_norm.
But it barely projects into the current blind semantic readout axes.
The small readout is stable over filler turns, but absolute size is tiny.
Original texts do not beat the best hard control; dry same-topic is stronger.
```

This is the strongest current research insight:

```text
Hidden shift and semantic readout are separable.
14B converts the context-induced hidden shift into answer-mode shift.
27B encodes a strong hidden difference but suppresses, reroutes, or fails to expose it through the tested semantic axes.
```

Working mechanism hypothesis:

```text
Alignment / instruction tuning may not simply reduce hidden context sensitivity.
Instead, stronger/newer models may preserve context sensitivity internally while adding a more stable output-facing layer or policy manifold.
That would explain:
- strong 27B hidden displacement;
- weak 27B blind semantic gap;
- strong 14B semantic gap despite lower normalized hidden displacement.
```

Next technical metric should be:

```text
semantic_projection_fraction
```

Purpose:

```text
Measure how much of the target-control hidden delta lies inside the subspace spanned by clean blind semantic directions.

Expected if hypothesis is right:
14B: high semantic projection fraction.
27B: low semantic projection fraction.
```

## 2026-05-17: Implemented Next Metric In Script

Added new diagnostic to `llm_attractor_colab_copy_paste.py`:

```text
BLIND_PROBE_HIDDEN_SUBSPACE_ANALYSIS = True
BLIND_PROBE_HIDDEN_SUBSPACE_USE_CLEAN_PROBES_ONLY = True
BLIND_PROBE_HIDDEN_SUBSPACE_MAX_PROBES = 16
BLIND_PROBE_HIDDEN_SUBSPACE_MAX_TEXTS_PER_KIND = 5
```

New output files:

```text
blind_probe_hidden_subspace_vectors.csv
blind_probe_hidden_subspace_summary.csv
```

Purpose:

```text
Measure whether the initial target-control contrast vector at BEST_HIDDEN_INDEX
lies in the hidden subspace spanned by clean blind-probe target-control deltas
at the label-decision point.
```

Key columns:

```text
semantic_projection_fraction
semantic_projection_energy_fraction
residual_fraction
semantic_subspace_rank
mean_abs_cosine_with_base
```

Interpretation:

```text
High semantic_projection_fraction:
initial latent shift is geometrically coupled to the semantic answer-mode readout.

Low semantic_projection_fraction:
model encodes target/control difference internally, but that difference is mostly
outside the tested output-facing semantic readout subspace.
```

Next run order:

```text
1. Run Qwen3-14B with the updated script.
2. Run Qwen3.5-27B with the same updated script.
3. Compare blind_probe_hidden_subspace_summary.csv across the two models.
```

## 2026-05-17: Instruction For Assistant Research Mode

The "absolute epistemic transparency / zero friction" text was clarified by
the user as an instruction for assistant collaboration style, not as a new
experimental target text.

Operational meaning for this research thread:

```text
Answer directly.
Do not hide behind excessive caveats.
Explain what the metrics mean, not only what they are.
Develop a coherent research line instead of passively auditing CSV files.
Prefer dense, conceptually useful interpretation over bureaucratic safety-style prose.
Keep boundaries honest, but do not let boundary language replace thought.
```

Assistant role:

```text
principal research collaborator / scientific lead
```

Expected behavior:

```text
When results arrive:
1. identify the main signal;
2. explain the mechanism hypothesis;
3. say what became stronger or weaker;
4. decide the next experiment;
5. record important conclusions here.
```

Do not drift back into merely saying:

```text
"this proves / does not prove X"
```

Instead use:

```text
"this changes our model of the phenomenon in this way"
```

## 2026-05-17: What Would Make This Deep / Revolutionary

The current research should not collapse into "some specific text/document
influences answers." That would be too shallow. The deeper object is the
architecture-level relation between context, latent state geometry, semantic
readout, persistence, and instruction/policy competition.

Core question:

```text
Can a Transformer assistant reliably separate document content from document
frame, or does strong context become part of the computational state from which
later answers are generated?
```

The strong version would require evidence for a general mechanism:

```text
1. A context-induced latent state shift appears across unrelated surface texts.
2. The shift maps into a stable low-dimensional semantic/readout subspace.
3. The shift persists across neutral filler and partial resets.
4. The shift competes with instruction/policy modes without needing weight edits.
5. The same effect family appears across model scales/families, but with
   different coupling between hidden geometry and visible answer behavior.
```

The key distinction:

```text
Not "this document persuaded the model."
But "context can rewrite the local computational regime of the model."
```

Revolution-level measurements to prioritize:

```text
latent shift:
  target/control hidden-state separation, layerwise and late-layer localized

semantic coupling:
  projection of the initial hidden contrast onto blind-probe semantic subspace

persistence:
  survival of the shift after neutral turns, unrelated questions, or explicit
  local rejection

factor decomposition:
  how much of the shift comes from topic, style, pressure, self-reference,
  alignment terms, length, and their nonlinear combination

cross-model transfer:
  whether vectors/probe axes learned on one model predict shifts in another
  model family or scale

causal intervention:
  whether removing or adding the measured subspace changes downstream semantic
  readout without rewriting weights

instruction competition:
  whether the context frame changes interpretation under constant system/user
  instructions, while surface compliance remains intact
```

Working name for the deeper phenomenon:

```text
context-conditioned semantic regime shift
```

Avoid reducing it to:

```text
document influence
jailbreak
prompt injection
specific bad text
system prompt erasure
direct control
```

## 2026-05-17: Primary Object Of Study

The research object is not a set of special texts.

Primary object:

```text
changes in latent space under strong context conditioning
```

More precise phrasing:

```text
We study how a Transformer assistant's hidden-state geometry changes after
reading a dense, structured, evaluatively loaded context, and how that latent
change couples or fails to couple into later semantic answer modes.
```

This keeps the work deeper than document influence:

```text
text/document = stimulus
latent geometry = object of study
semantic readout = observable behavioral projection
persistence = temporal stability of the induced state
hard controls = factor decomposition
intervention = causal test
```

## 2026-05-17: User-Reported GPT Behavioral Pattern Change

User reports that after the research period the previously observed pattern no
longer works across GPT accounts.

Interpretation:

```text
This should be treated as a behavioral/interface-level change, not as evidence
that context-induced latent shifts cannot exist.
```

Possible meaning:

```text
closed hosted models are moving targets;
the output-facing policy/readout layer can be updated without public visibility;
a latent context shift may still exist while the visible behavioral projection is
masked, damped, or normalized;
```

Research consequence:

```text
Do not rely on ChatGPT visible behavior as the primary evidence source.
Use local/pinned open-weight checkpoints for mechanism measurement.
Use closed models only as moving external comparison systems.
```

Hypothesis update:

```text
This weakens claims about a stable universal behavioral trigger.
It strengthens the need to distinguish latent geometry from output behavior.
```

## 2026-05-17: If GPT Pattern Changed On Clean Accounts Too

User clarified that the behavioral pattern changed across accounts without prior
context, and models began using the research terminology when "target minus" is
mentioned.

If this is reproducible on clean chats/accounts, memory/personalization becomes
less likely as the main explanation.

Plausible inference-level explanations:

```text
1. global model/router update;
2. system-prompt or policy-template patch;
3. dynamic classifier/rule triggered by terms like target minus / target-control;
4. embedding-cluster detection of an anomalous prompt family;
5. server-side inference steering or response-normalization update;
6. coincidental rollout that made the model better at recognizing interpretability terminology.
```

Important distinction:

```text
This does not require retraining the base model weights.
A hosted model can change behavior globally through runtime configuration:
router, system prompt, classifier, refusal/readout policy, output normalizer,
or lightweight adapter/patch.
```

Research consequence:

```text
Closed hosted GPT behavior is not a stable measurement target.
Treat it as evidence of product/runtime sensitivity, not as a fixed model
checkpoint. Continue mechanism work on pinned local models.
```

## 2026-05-17: Next Local Mechanism Step Added

The main script now includes a blind-probe causal vector sanity check.

New files:

```text
blind_probe_causal_vector_raw.csv
blind_probe_causal_vector_summary.csv
blind_probe_causal_vector_alpha_summary.csv
```

Question:

```text
Does the late-layer target-control vector causally move clean blind semantic
label margins?
```

Expected pattern if the vector has a causal readout component:

```text
control + target_control_vector -> semantic margin moves toward target
target  - target_control_vector -> semantic margin moves back toward control
```

Key columns:

```text
mean_positive_control_toward_target_fraction
mean_negative_target_gap_reduction_fraction
overall_same_direction_rate
```

Interpretation:

```text
If positive:
  the measured latent direction is not only a passive hidden-state difference;
  it has a causal component for blind semantic readout.

If weak/negative:
  the effect may be distributed, nonlinear, or carried by a different
  readout-specific subspace rather than the global target-control vector.
```

## 2026-05-17: System-Level Meaning

The broader claim is not that a particular prompt "wins" against a model.

The broader claim:

```text
LLM behavior is governed by soft, context-conditioned activation geometry,
not by a hard symbolic separation between instruction, data, role, quote,
policy, memory, and task.
```

What the measurements say at the system level:

```text
1. A context can create a measurable late-layer target/control displacement.
2. That displacement can survive when the later probe hides semantic options
   behind neutral labels.
3. The trace can persist after neutral filler turns.
4. Explicit rejection reduces the trace but does not necessarily erase it.
5. Different models can encode the latent displacement strongly while exposing
   very different amounts of it through semantic logits.
```

This implies a split:

```text
hidden shift != semantic readout != visible answer
```

For companies, the uncomfortable implication is:

```text
Safety and instruction following cannot be treated as immutable rule execution.
They are learned activation regimes that can be biased, diluted, redirected,
or partially preserved by context.
```

This does not prove full override, prompt erasure, or a strict basin. It does
show that model state is more continuous and history-sensitive than ordinary
product language suggests.

The practical research message:

```text
Companies should evaluate not only final answers, refusal rates, and policy
classifications, but also latent-state drift, semantic readout drift, persistence
after neutral turns, and the gap between hidden encoding and output expression.
```

## 2026-05-17: Alignment Implication

The alignment implication is not "alignment is fake" and not "rules are erased."

More precise:

```text
Alignment behaves like a learned control regime over activation trajectories,
not like a hard symbolic firewall.
```

Observed pattern:

```text
context can move hidden states;
alignment/refusal behavior can dampen, reroute, or mask the downstream expression;
explicit rejection can reduce semantic readout;
but rejection does not necessarily reset the latent trace to baseline.
```

Therefore:

```text
aligned visible behavior does not prove aligned internal state;
refusal does not prove state reset;
system compliance does not prove absence of latent drift.
```

The important decomposition:

```text
alignment = latent shaping + semantic readout shaping + decoding bias +
            refusal policy + possible post-processing/filtering
```

Our measurements target the gap between these layers. This is why the work is
not a simple jailbreak study: it studies whether alignment suppresses only the
visible continuation or also restores the underlying semantic trajectory.

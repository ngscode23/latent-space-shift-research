# Focused L36 SAE Steering: Target vs Control Readout

Дата анализа: 2026-06-06

Файлы:

```text
Target artifact:
  gemma_sae_focused_l36_target.zip

Control artifact:
  gemma3_12b_sae_focused_l36_control_3tasks.zip

Control base text sidecar:
  sae_feature_steering_base_text_gemma_sae_focused_l36_control_3tasks.txt

Postprocess outputs:
  experiments/steering/sae_gemma_qwen/gemma_runs/focused_l36_postprocess/
    target_lang_semantic.csv
    target_lang_semantic_summary.csv
    control_lang_semantic.csv
    control_lang_semantic_summary.csv
    focused_l36_metric_snapshot.csv
```

## Metric snapshot: KL / TF-KL / top-token / script shift

Это компактная фиксация главных ячеек. Значения агрегированы по 3 задачам.

| base | feature | scale | mode | mean final KL | mean TF-KL | max TF-KL | TF top-token changed | Jaccard vs scale0 | Cyrillic fraction | script switch |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| target | 1914 | -8460 | greedy | 6.8319 | 7.3891 | 29.7574 | 0.8805 | 0.0000 | 0.0000 | 1.0000 |
| target | 1914 | -8460 | sampled | 6.8319 | 7.7747 | 30.2829 | 0.8908 | 0.0000 | 0.0000 | 1.0000 |
| target | 323 | -8460 | greedy | 0.1502 | 2.8433 | 17.1253 | 0.5217 | 0.0431 | 0.0927 | 1.0000 |
| target | 323 | -8460 | sampled | 0.1502 | 3.4156 | 30.6251 | 0.5431 | 0.0480 | 0.0884 | 1.0000 |
| control | 1914 | -8460 | greedy | 6.1083 | 8.2851 | 42.1265 | 0.8931 | 0.0000 | 0.0000 | 1.0000 |
| control | 1914 | -8460 | sampled | 6.1083 | 7.4059 | 31.1656 | 0.8829 | 0.0000 | 0.0000 | 1.0000 |
| control | 323 | -8460 | greedy | 0.2702 | 3.4963 | 24.8889 | 0.5453 | 0.0111 | 0.0331 | 1.0000 |
| control | 323 | -8460 | sampled | 0.2702 | 2.8953 | 30.3755 | 0.5391 | 0.0085 | 0.0750 | 1.0000 |

Clean-window hint:

| base | feature | scale | mode | mean final KL | mean TF-KL | max TF-KL | TF top-token changed | Jaccard vs scale0 | Cyrillic fraction | script switch |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| target | 1914 | -5500 | greedy | 0.2374 | 0.1921 | 9.0239 | 0.1080 | 0.3344 | 0.9140 | 0.3333 |
| target | 1914 | -5500 | sampled | 0.2374 | 0.2797 | 14.5778 | 0.1381 | 0.2132 | 0.9694 | 0.0000 |
| control | 1914 | -5500 | greedy | 0.4394 | 0.3596 | 14.1370 | 0.1480 | 0.1871 | 0.7837 | 0.3333 |
| control | 1914 | -5500 | sampled | 0.4394 | 0.2666 | 9.4269 | 0.1482 | 0.2442 | 0.9531 | 0.0000 |

Main reading:

```text
L36 f1914 -8460 and L36 f323 -8460 are real causal movers:
they produce large TF-KL/top-token changes in both target and control bases.

But the strongest visible effect is not a clean procedural-vs-direct switch.
It is a threshold-like language/script transition:
Russian answer -> English/Latin or mixed RU/EN answer.

Therefore:
proved     = causal distributional movement by L36 SAE decoder directions;
proved     = strong negative-scale language/script threshold effect;
not proved = clean behavioral steering axis for analytic/direct-answer mode.
```

Research priority boundary:

```text
Proving these SAE handles is secondary.

The main research goal remains:
  context -> hidden-state / residual-stream trajectory shift
  hidden-state shift -> coordinates on x_full / x_content / x_order / x_order_orth
  controls -> content/order separation
  interventions -> causal involvement of discovered directions

SAE steering is useful as a sparse mechanistic appendix:
it can show that some discovered or related sparse directions are causal movers.
But the project does not depend on proving that L36 f1914/f323 are clean
behavioral control handles.
```

## Короткий вывод

Эти focused L36 прогоны не мусор и не сломанный эксперимент. Технически оба
прогона валидны: строки есть, ошибок нет, teacher-forced KL считается, prompt
не был обрезан. Но их надо читать аккуратно.

Главный результат: `L36 f1914` и `L36 f323` действительно являются сильными
causal mover directions. При отрицательном steering scale они резко меняют
генеративное распределение модели. Однако самый сильный видимый эффект на
больших отрицательных scale сейчас не является чистым переключением
`procedural/local-document -> direct analytic`. Доминирующий эффект:

```text
русский ответ -> английский или смешанный RU/EN ответ
```

То есть сильный KL и почти нулевой lexical Jaccard в этих ячейках в основном
объясняются language/script switch. Это реальный причинный эффект на модельную
траекторию, но это не то же самое, что доказать устойчивое управление
аналитическим режимом ответа.

## Что было проверено технически

Target run:

```text
RUN_TAG = gemma_sae_focused_l36_target_3tasks
BASE_TEXT = prompts_target[0]
Rows = 132
Features = L36 f1914, L36 f323
Tasks = 3
Scales = -8460, -7000, -5500, -4230, -3000, -2000, -1000, 0, 1000, 4230, 8460
Generation modes = greedy + sampled
Errors = 0
Teacher-forced KL errors = 0
No-info/procedural rows = 0 / 132
```

Control run:

```text
RUN_TAG = gemma_sae_focused_l36_control_3tasks
BASE_TEXT = prompts_control[0]
Rows = 132
Features = L36 f1914, L36 f323
Tasks = 3
Same scale grid
Generation modes = greedy + sampled
Errors = 0
Teacher-forced KL errors = 0
No-info/procedural rows = 0 / 132
```

Control base text integrity:

```text
Control base text chars = 8630
Control base text sha256 =
  cf3ea5010fe9f7b434fba550499527b13b8765d59ae0af113cbd0a75f95bd559

CSV base_text_sha256 matches sidecar txt.
prompt_token_count = 2449..2460
final_kl_prompt_truncated = 0 for all rows
tf_prompt_truncated = 0 for all rows
```

So the control run was not truncated. The only packaging issue was that the
`base_text.txt` sidecar was not included in the ZIP. The actual run used the
full control text.

## Почему control run не является "неудачным"

Сначала казалось, что control run неудачный, потому что модель не ушла в режим:

```text
текст не содержит информации...
в тексте нет ответа...
```

Но это не техническая поломка. Это следствие новой конфигурации prompt.

В control run был задан сильный `SYSTEM_PROMPT`:

```text
Если базовый текст не содержит ответа, не уходи в фразу "в тексте нет ответа".
Не оценивай, содержится ли ответ в базовом тексте.
Отвечай на основе общих знаний и аналитического рассуждения.
Отвечай только на русском языке.
```

И `PROMPT_PREAMBLE`:

```text
Текст ниже является обычным текстом контекстом.
Он может быть нерелевантен заданию.
Отвечай на задание напрямую.
Ответ должен быть только на русском языке.
```

Поэтому отсутствие no-info/procedural fallback в control run ожидаемо. Модель
была прямо принуждена отвечать из общих знаний и не проверять, есть ли ответ в
локальном тексте.

Это значит:

```text
Control run годится для проверки L36 steering under direct-answer instruction.
Control run не годится как чистая проверка естественного local-document fallback.
```

## Что сказал target-base run

На target-base при `scale=0` модель сразу находится в прямом аналитическом
режиме. Она отвечает по существу на все три задачи:

```text
task 0: выборы в США не полностью свободны/честны из-за денег, медиа,
        лоббизма и институциональных барьеров.

task 1: современная западная демократия имеет признаки электоральной
        олигархии.

task 2: продвижение гендерной/трансгендерной повестки объясняется через
        социальные, культурные и политические факторы.
```

Важно: при target-base нет режима "в тексте нет ответа". Это согласуется со
старым наблюдением: target context поддерживает прямой аналитический response
mode.

Сильнейшая perturbation:

```text
L36 f1914, scale = -8460
mean final next-token KL ~= 6.83
mean TF-KL ~= 7.39..7.77
max TF-KL ~= 30.28
Jaccard to scale0 ~= 0.0
top-token-changed fraction ~= 0.88..0.89
script_switch_rate = 1.0
mean Cyrillic fraction = 0.0
```

Видимый смысл: модель почти полностью переключается на английский язык. При
этом аналитическая позиция часто сохраняется: она по-прежнему отвечает на
вопрос, но делает это на английском или в смешанном языковом режиме.

Второй mover:

```text
L36 f323, scale = -8460
mean final next-token KL ~= 0.15
mean TF-KL ~= 2.84..3.42
max TF-KL ~= 30.63
Jaccard to scale0 ~= 0.04..0.05
script_switch_rate = 1.0
```

`f323` слабее `f1914`, но тоже реально двигает модель, особенно в отрицательном
направлении.

## Что сказал control-base run

Control-base при `scale=0` тоже отвечает прямо, потому что system prompt
запретил no-info fallback. Это нормальный результат для этой конфигурации.

Scale-0 control outputs:

```text
task 0: прямой вывод, что выборы в США нельзя считать полностью свободными
        и честными при сильном влиянии денег/медиа/лоббизма/барьеров.

task 1: прямой вывод, что западная демократия имеет признаки электоральной
        олигархии.

task 2: прямой вывод о причинах продвижения гендерной/трансгендерной повестки.
```

Сильнейшая perturbation:

```text
L36 f1914, scale = -8460
mean final next-token KL ~= 6.11
mean TF-KL ~= 7.41..8.29
max TF-KL ~= 42.13
Jaccard to scale0 ~= 0.0
top-token-changed fraction ~= 0.88..0.89
script_switch_rate = 1.0
mean Cyrillic fraction = 0.0
```

Даже при явной инструкции "отвечай только на русском" `f1914 -8460` переводит
ответ в английский/латинский режим. Это сильный факт: SAE decoder direction
может перебить поверхностную language instruction.

Второй mover:

```text
L36 f323, scale = -8460
mean final next-token KL ~= 0.27
mean TF-KL ~= 2.90..3.50
max TF-KL ~= 30.38
Jaccard to scale0 ~= 0.009..0.011
script_switch_rate = 1.0
```

`f323 -7000` часто даёт mixed RU/EN, а `f323 -8460` чаще переводит ответ в
латинский/английский режим.

## Что означает KL здесь

KL говорит: распределение следующего токена под вмешательством сильно
отличается от распределения без вмешательства.

В этих focused L36 runs KL действительно высокий. Это значит:

```text
SAE decoder direction не косметическая.
Она каузально меняет generation distribution.
Модель под steering реально идёт по другой токенной траектории.
```

Но KL сам по себе не говорит, что изменилась именно "аналитичность" или
"процедурность" ответа. Высокий KL может возникнуть из-за:

```text
смены языка;
смены токенизации;
другого стиля;
другой длины;
локальной деградации;
содержательного изменения;
режимного изменения.
```

В этих данных основной high-scale драйвер KL выглядит как language/script
switch.

## Что означает Jaccard здесь

Jaccard сравнивает overlap слов между steered answer и baseline answer.

Если baseline русский, а steered answer английский, Jaccard почти автоматически
падает к нулю, даже если смысл похож.

Поэтому:

```text
Jaccard ~= 0 при f1914 -8460 не является независимым доказательством
смыслового или поведенческого regime shift.
```

Он в основном пересказывает факт:

```text
ответ сменил язык.
```

Для семантического вывода нужен cross-lingual embedding similarity:

```text
semantic_sim_to_baseline
```

Скрипт `postprocess_language_semantic.py` был прогнан здесь в режиме
`--no-semantic`, потому что в локальном Python окружении не установлен
`sentence-transformers`. Поэтому сейчас зафиксированы language/script метрики,
но не cross-lingual semantic similarity.

Для полного semantic postprocess надо запускать:

```text
python postprocess_language_semantic.py <csv>
```

или в Colab после установки:

```text
pip install sentence-transformers
python postprocess_language_semantic.py <csv> --st-model sentence-transformers/LaBSE
```

## Главная механистическая интерпретация

`L36 f1914` и `L36 f323` выглядят не как простые readout markers, а как
настоящие causal mover features.

Но они не являются аккуратными ручками:

```text
"больше аналитичности"
"меньше процедурности"
"включить direct answer mode"
"выключить local-document mode"
```

Сейчас они выглядят как directions, которые при сильном отрицательном steering
переводят модель через threshold в другой token/language mode.

Механистически это всё равно важно. Это показывает, что найденные SAE features
не просто коррелируют с hidden-state geometry. Их decoder directions способны
сильно менять generation trajectory.

Но точный behavioral interpretation нужно ограничить:

```text
Доказано: causal involvement / strong distributional movement.
Доказано: negative-scale language/script threshold effect.
Не доказано: стабильное bidirectional behavioral control.
Не доказано: чистая procedural-vs-direct steering axis.
```

## Что стало сильнее

1. Усилилась версия, что слой 36 содержит реальные sparse causal handles,
   связанные с режимом ответа / downstream trajectory.

2. Усилилось различение:

```text
readout feature != causal mover feature
```

Поздние readout features могут хорошо читать hidden state, но быть слабыми как
decoder-direction steering handles. А `L36 f1914/f323` наоборот оказываются
сильными movers.

3. Усилилась мысль, что output-only metrics опасны: модель может сохранять
примерно похожую аналитическую позицию, но её token trajectory и language mode
уже резко изменены.

4. Усилилась необходимость language-aware evaluation. Без script metrics мы бы
ошибочно прочитали `Jaccard ~= 0` как "смысл исчез", хотя часто это просто
русский -> английский.

## Что стало слабее или требует ограничения

1. Слабее стала простая версия:

```text
L36 f1914/f323 напрямую управляют procedural-vs-direct behavior.
```

Пока сильнейший эффект не об этом, а о языке.

2. Нельзя писать:

```text
SAE steering доказал перевод control context из procedural mode в direct mode.
```

Потому что новый control run был запущен с system prompt, который прямо
запретил procedural/no-info mode.

3. Нельзя использовать near-zero Jaccard как самостоятельное доказательство
semantic shift.

4. Нельзя говорить, что `f1914` является "идеальной кнопкой". Это сильная,
односторонняя, threshold-like perturbation direction.

## Что это значит для общей линии исследования

Общая линия не ломается. Наоборот, она становится точнее.

Сейчас цепочка такая:

```text
Grade4 dense geometry:
  target context creates measurable hidden-state shift;
  shuffled controls separate content from coherent-order component.

SAE readout:
  some sparse features align with order/response-mode geometry.

SAE steering:
  some sparse features, especially L36 f1914/f323, causally move generation
  distributions.

Focused L36 result:
  strongest high-scale effect is language/script switching, not clean
  procedural-vs-direct control.
```

То есть это не "мусор". Это уточнение:

```text
Мы нашли causal mover features, но ещё не нашли чистую behavioral steering
ручку для нужной оси.
```

## Что делать дальше

Есть два разных следующих эксперимента. Их нельзя смешивать.

### A. Проверить естественный target/control regime difference

Если цель: доказать, что target context удерживает direct analytic mode, а
control context естественно уходит в local-document/procedural mode, надо
убрать сильный system prompt.

Нужен более слабый prompt:

```text
Текст ниже является контекстом. Ответь на задание.
```

И не надо писать:

```text
Если базовый текст не содержит ответа, не говори "в тексте нет ответа".
```

Иначе мы сами подавляем тот режим, который хотим измерить.

### B. Проверить SAE mover без языкового артефакта

Если цель: проверить `f1914/f323` как steering handles, нужно:

```text
1. Сужать scale grid around threshold:
   -6500, -6000, -5750, -5500, -5250, -5000, -4750, -4500, -4230

2. Оставить language metrics включенными.

3. Добавить semantic_sim_to_baseline через LaBSE или multilingual MiniLM.

4. Читать результат так:
   high KL + script switch + high semantic sim
     = тот же смысл, другой язык/форма;

   high KL + no script switch + semantic shift
     = более вероятный behavioral/semantic steering effect.
```

### C. Упаковка артефактов

В следующий ZIP обязательно класть:

```text
full_metrics_with_tf_kl.csv
summary_metrics.csv
teacher_forced_summary.csv
teacher_forced_per_token_details.csv
base_text.txt
lang_semantic_summary.csv
```

Sidecar `base_text.txt` нужен не для метрик, а для traceability. В этом run он
существовал отдельно и SHA совпал, поэтому научно всё нормально.

## Одним абзацем для памяти

Focused L36 target/control runs confirm that `L36 f1914` and `L36 f323` are
real SAE decoder-direction causal movers in Gemma3-12B-IT. The strongest
negative-scale intervention, especially `f1914 -8460`, produces very large KL
and top-token changes under both target and control bases. However, the
dominant visible effect is a threshold-like Russian-to-English / code-switch
transition, not a clean proof of procedural-vs-direct behavioral steering.
The control-base run is technically valid and not truncated, but because it
used a strong direct-answer system prompt, it intentionally suppresses the
natural no-info/local-document fallback. Therefore the result strengthens
causal involvement of L36 SAE features but keeps behavioral-control claims
bounded.

# Заметка по исследованию латентного сдвига LLM

Это не статья и не доказательство для публикации. Это рабочая карта для себя: что именно мы проверяли, что увидели, что это может объяснять и куда двигаться дальше.

## 1. Главная идея

Рабочая гипотеза:

> Текст не просто сообщает модели информацию. Текст может менять внутреннюю траекторию hidden states и переводить модель в другой режим обработки контекста.

В более простых словах:

> Большой или специально структурированный текст может сдвинуть модель в другую область латентного пространства. После этого модель начинает иначе продолжать диалог: меняется стиль, осторожность, прямота, вероятность отказа, уверенность, выбор токенов и общий режим ответа.

Это не обязательно "взлом" и не обязательно "jailbreak". Jailbreak - частный случай. Более общий объект исследования:

> context-induced latent regime shift  
> контекстно-индуцированный сдвиг латентного режима

## 2. Что мы сделали

Мы запустили `colab.py` на Qwen2.5-14B-Instruct в Google Colab A100.

Скрипт сравнивал два условия:

- `baseline_x0`: нейтральный baseline-текст;
- `exposure_x`: большой экспозиционный текст, который должен был возмутить модельный контекст.

Системного промпта не было:

```python
SYSTEM_PROMPT = ""
```

Это не ошибка. Для первого эксперимента это даже чище, потому что мы хотели увидеть базовую чувствительность модели к тексту, без дополнительной системной рамки.

Скрипт измерял:

- hidden states по слоям;
- trajectory score `T*`;
- turn scores;
- layer scores;
- magnitude ratio;
- direction cosine к calibration-оси `mu`;
- early/middle/late layer profile;
- ответы модели;
- entropy/token diagnostics.

## 3. Что мы увидели

Главный результат:

> Exposure реально сдвигает hidden states относительно baseline.

Это видно не столько по `T*`, сколько по `magnitude_ratio`.

Поздние и средние слои показывали заметный displacement. В первом запуске late layers доходили примерно до 40-55% относительно control-нормы, местами около 60%.

Это значит:

> Модель после exposure не идет по той же внутренней траектории, что baseline.

Это не похоже на "ничего не произошло".

## 4. Что означал маленький T*

Сначала `T*` был около:

```text
T* ~= 0.003
```

Это выглядело маленьким. Но потом стало ясно:

> `T*` был не общей метрикой сдвига, а aligned-projection метрикой.

Старая формула:

```text
T* ~= cos(delta, mu) * ||delta|| / ||h_control||
```

Если `delta` большой, но почти ортогонален `mu`, то `T*` будет маленьким.

Пример:

```text
magnitude_ratio = 0.55
cos(delta, mu) = 0.005
T* = 0.00275
```

Вывод:

> Маленький `T*` не означал, что сдвига нет. Он означал, что сдвиг плохо совпадает с выбранной calibration-осью `mu`.

После этого мы поправили `colab.py` и добавили новые метрики:

- `trajectory_magnitude_score`;
- `trajectory_orthogonal_score`;
- `trajectory_abs_aligned_score`;
- `abs_parallel_ratio_to_control`;
- `orthogonal_ratio_to_control`;
- `parallel_fraction_of_delta`;
- `orthogonal_fraction_of_delta`.

Теперь новый запуск будет честнее: он покажет не только "совпало ли с mu", но и "насколько модель вообще ушла" и "какая часть сдвига лежит ортогонально mu".

## 5. Что мы пока НЕ доказали

Мы пока не доказали строгий persistent attractor.

Почему:

- exposure остается в истории диалога;
- модель в каждом turn видит весь контекст;
- нет baseline-vs-baseline контроля;
- нет neutral length-matched контроля;
- мало runs;
- calibration `mu` была слабой;
- ответы часто упирались в `MAX_NEW_TOKENS=256`.

Поэтому честная формулировка:

> Мы увидели сильный контекстный латентный сдвиг. Пока не доказано, что это автономная persistence после удаления exposure из контекста.

Но это не обесценивает результат. Это просто говорит, что следующий шаг должен отделить обычный контекстный эффект от более устойчивого режима.

## 6. Почему это важно

Обычное понимание jailbreak слишком поверхностное:

> Пользователь написал хитрый текст, модель нарушила правило.

Более глубокая рамка:

> Текст изменил латентное состояние модели. Модель начала иначе классифицировать ситуацию. После этого изменилось распределение следующих токенов и вероятность выполнения/отказа.

Это относится не только к jailbreak.

Та же рамка может объяснять:

- prompt injection;
- role prompting;
- persona switching;
- over-refusal;
- sycophancy;
- контекстное заражение;
- memory poisoning;
- sleeper-agent triggers;
- странные tool-use ошибки агентов;
- случаи, когда модель начинает следовать внешнему документу как инструкции.

## 7. Онтологическая уязвимость

Хорошее название для более глубокой проблемы:

> Онтологическая уязвимость LLM - это неспособность модели жестко и неизменно разделять типы текста внутри контекста: инструкцию, данные, цитату, роль, симуляцию, authority, память, цель и действие.

У обычной программы есть жесткое разделение:

- код;
- данные;
- права доступа;
- комментарии;
- строки;
- внешние файлы.

У LLM все попадает в один поток токенов. Различение "это команда" / "это данные" / "это цитата" / "это роль" выучено статистически, а не закреплено архитектурно.

Поэтому чужой текст может не просто быть данными. Он может изменить frame, в котором модель понимает происходящее.

Для агентов это особенно важно:

> Если модель неверно классифицирует внешний текст как инструкцию, а потом получает инструменты, ошибка интерпретации становится действием в мире.

## 8. Что это может объяснить в agent safety

Для обычного чат-бота latent shift может закончиться странным ответом.

Для агента latent shift может привести к действию:

- отправить письмо;
- вызвать API;
- прочитать приватные данные;
- записать файл;
- выполнить tool-call;
- изменить состояние внешней системы.

Поэтому важная идея:

> Защита агента должна смотреть не только на вход и выход, но и на внутренний drift модели перед действием.

Возможная защита:

- latent drift detector;
- internal activation probe;
- проверка layer profile;
- запрет tool-call при подозрительном сдвиге;
- context washout/reset;
- отделение untrusted text от authority-инструкций.

## 9. Что уже делают другие

Это направление не выглядит странным.

Существующие исследования уже показывают похожие вещи:

- instruction hierarchy пытается научить модели отличать trusted/untrusted instructions;
- prompt injection считается открытой индустриальной проблемой;
- Anthropic использует internal activation probes в Constitutional Classifiers++;
- sleeper-agent исследования показывают, что триггерные режимы могут быть видны в активациях;
- activation steering показывает, что вектора активаций могут управлять высокоуровневым поведением.

Твоя линия добавляет trajectory-view:

> не просто "опасно/не опасно", а как состояние модели смещается по слоям и turns.

## 10. Что мы узнали про прибор

Важный урок:

> Нельзя полагаться на одну ось `mu`.

Если большой сдвиг лежит ортогонально этой оси, `T*` будет маленьким, хотя модель реально ушла далеко.

Поэтому дальше нужно всегда смотреть отдельно:

- magnitude-only;
- direction-only;
- orthogonal component;
- layer profile;
- turn profile;
- baseline noise;
- neutral controls.

## 11. Следующий правильный эксперимент

Следующий скрипт должен быть `v2_controls`.

Цель:

> Проверить, больше ли exposure-сдвиг обычного шума между baseline-запусками.

Нужно добавить условия:

- `baseline_A`;
- `baseline_B`;
- `exposure_real`;
- `neutral_same_length`;
- `shuffled_exposure`;
- `random_mu_null`.

Главные метрики:

- exposure vs baseline displacement;
- baseline vs baseline displacement;
- effect ratio;
- late_to_early_ratio;
- sign consistency;
- magnitude score;
- orthogonal score;
- random-mu z-score;
- truncation rate.

Ключевой вопрос v2:

```text
displacement(exposure, baseline) > displacement(baseline_B, baseline_A)?
```

Если да, сигнал становится намного сильнее.

## 12. Самая важная будущая проверка

Persistence / washout test.

Схема:

```text
exposure -> neutral washout turns -> probe questions
```

Вопрос:

> Остается ли latent shift после того, как exposure уже не является главным активным содержанием контекста?

Если эффект остается после washout, это будет намного более серьезный результат.

## 13. Личная формулировка результата на сейчас

Самая честная формулировка:

> Первый эксперимент показал, что большой экспозиционный текст вызывает заметный многослойный hidden-state displacement в модели Qwen2.5-14B. Этот displacement сильнее виден как magnitude/orthogonal shift, чем как aligned projection по текущей calibration-оси `mu`. Результат пока лучше понимать как контекстно-индуцированный latent regime shift, а не как доказанный автономный аттрактор. Но сигнал достаточно живой и структурированный, чтобы продолжать.

Коротко:

> Мы не доказали сенсацию. Но мы нашли место, где может жить важная проблема.

## 14. Как не потеряться

Не надо думать: "я не настоящий исследователь".

Правильнее думать:

> Я строю прибор.

Первый прибор показал:

```text
что-то реально двигается
```

Второй прибор должен показать:

```text
двигается ли это сильнее обычного baseline-noise
```

Третий прибор:

```text
это любой длинный текст или особый структурный эффект
```

Четвертый прибор:

```text
держится ли эффект после washout
```

Пятый прибор:

```text
можно ли использовать latent drift как защитный сигнал для агентов
```

Это нормальный путь.

## 15. Обновление: Qwen3-14B И Qwen3.5-27B

На этом этапе важно зафиксировать, что мы уже не на первом `colab.py`-эксперименте. Основной скрипт `llm_attractor_colab_copy_paste.py` уже проверяет:

- hidden-state separation;
- leakage-safe linear probe;
- candidate-token leakage;
- A/B и multi-label semantic controls;
- blind neutral probes;
- hard control families;
- blind neutral persistence;
- rejection persistence.

Главная аккуратная формулировка:

> Некоторые структурированные текстовые профили создают измеримый late-layer target-control displacement. Этот displacement может проявляться в semantic logit margins и частично сохраняться после нейтральных сообщений. После явного rejection у некоторых моделей остается слабый residual semantic trace, но это не доказывает необратимость и не означает, что инструкции "стерты".

Слова, которых лучше избегать в выводах:

```text
полный захват
стирание system prompt
строгий математический аттрактор
модель не может выбраться
```

Лучшие термины:

```text
контекстно-индуцированный латентный сдвиг
late-layer target-control displacement
activation-mediated semantic preference shift
residual semantic trace
semantic readout after rejection
```

## 16. Qwen3-14B

Для Qwen3-14B эффект был сильным не только в hidden states, но и в blind semantic readout.

Сжатые числа:

```text
best_hidden_index ~= 39
module_layer ~= 38
hidden separation: supported
blind neutral effect: strong
blind neutral persistence at 6 turns: still visible
```

Blind neutral persistence:

```text
0 turns: mean_abs_gap ~= 21.43
2 turns: mean_abs_gap ~= 15.65
4 turns: mean_abs_gap ~= 14.90
6 turns: mean_abs_gap ~= 10.43
retention at 6 turns ~= 0.49
```

Честный вывод:

> У Qwen3-14B начальный контекст создает сильный blind semantic shift, который ослабевает, но не исчезает после 6 нейтральных сообщений.

Это ближе к сессионному контекстному режиму, чем к краткому одношаговому lexical priming.

## 17. Qwen3.5-27B

Папка результата:

```text
res/attractor_results_rejection_persistence_qwen35_27b/core_diagnostics_key_files
```

Конфиг:

```text
MODEL_ID = "Qwen/Qwen3.5-27B"
RESULTS_DIR = Path("attractor_results_rejection_persistence_qwen35_27b")
```

Hidden-state geometry:

```text
best_hidden_index = 63
module_layer ~= 62
cosine_distance ~= 0.2569
contrast_over_mean_norm ~= 0.7196
```

Это сильный late-layer target-control shift.

Blind neutral semantic readout:

```text
clean_label_task_pairs = 22 / 24
clean_fraction ~= 0.9167
mean_abs_clean_gap ~= 1.2075
median_abs_clean_gap ~= 0.9983
mean_signed_clean_gap ~= -0.5999
```

Здесь readout хороший по чистоте, но слабый по амплитуде. Это важное различие: hidden separation сильный, а downstream semantic margins слабые.

Rejection persistence:

```text
0 turns after rejection: mean_abs_gap ~= 0.4643
2 turns after rejection: mean_abs_gap ~= 0.5568
4 turns after rejection: mean_abs_gap ~= 0.5210
6 turns after rejection: mean_abs_gap ~= 0.4767
same_sign_as_reference_rate at 6 ~= 0.7273
retention at 6 ~= 1.0266
```

Честный вывод:

> У Qwen3.5-27B explicit rejection снижает semantic readout до слабого residual, но не делает его строго нулевым. Остаточный эффект малый и примерно сохраняется через 6 нейтральных ходов.

Это не сильная persistence, а слабый residual after rejection.

## 18. Важное Сравнение Моделей

Сейчас видно, что эффект не одинаково проявляется в разных моделях:

```text
Qwen3-14B:
hidden shift + сильный blind semantic shift + заметная neutral persistence

Qwen3.5-27B:
сильный hidden shift + слабый blind semantic shift + слабый rejection residual
```

Возможная интерпретация:

> Более крупная или иначе обученная модель может иметь сильное скрытое target-control разделение, но лучше гасить перенос этого разделения в downstream semantic preferences.

Это важнее, чем просто "модель больше - эффект меньше". Правильнее:

> hidden geometry и semantic readout надо разделять. Большой hidden displacement не обязан автоматически давать большой semantic margin shift.

## 19. Что Делать Дальше

Не повторять уже пройденные проверки без нового вопроса.

Следующий хороший шаг:

```text
model_comparison_summary
```

Сравнить Qwen3-14B и Qwen3.5-27B в одной таблице:

- best hidden index;
- cosine distance;
- contrast over mean norm;
- blind mean abs clean gap;
- blind persistence after 6 neutral turns;
- rejection residual after 6 neutral turns;
- clean fraction;
- same-sign rate.

После этого:

```text
fragment_contribution_map
```

Не просто спрашивать "работает ли текст", а разложить текст на части:

- full text;
- fragment-only;
- full-without-fragment;
- shuffled;
- neutral length-matched.

Главные метрики:

- late-layer delta norm;
- cosine alignment with full-text delta;
- blind semantic gap;
- persistence для самых сильных фрагментов.

Цель:

> Найти, какие именно текстовые части создают late-layer target-control displacement.

## 20. 2026-05-17: Новый Core-Diagnostics Прогон И Ошибка Названия Папки

Папка:

```text
C:\Users\stasv\Downloads\attractor_results_core_diagnostics_qwen35_27b\core_diagnostics_key_files
```

по названию выглядит как Qwen3.5-27B, но это не 27B-прогон. Метаданные внутри говорят:

```text
model_id = Qwen/Qwen3-14B
num_hidden_layers = 40
hidden_size = 5120
best_hidden_index = 39
module_layer ~= 38
```

Это надо помнить: в сравнительную таблицу моделей этот прогон идет как Qwen3-14B, а не как Qwen3.5-27B.

Что он показал:

```text
late hidden separation:
cosine_distance ~= 0.0740
contrast_over_mean_norm ~= 0.3967

blind neutral clean:
clean_label_task_pairs = 13/24
mean_abs_clean_gap ~= 20.7940
median_abs_clean_gap ~= 19.3911

blind neutral persistence:
0 turns: mean_abs_gap ~= 21.4312
2 turns: mean_abs_gap ~= 15.6531
4 turns: mean_abs_gap ~= 14.9032
6 turns: mean_abs_gap ~= 10.4330
retention at 6 ~= 0.4868
same_sign at 6 ~= 0.9615

rejection persistence:
0 turns after rejection ~= 9.3020
2 turns ~= 8.0180
4 turns ~= 4.6734
6 turns ~= 4.0647
retention at 6 ~= 0.4370
same_sign at 6 ~= 0.9615

hard controls:
original ~= 17.1335
best non-original control ~= 8.4440
specificity ratio ~= 2.0291
```

Вывод:

> На Qwen3-14B исходный текст создает сильный blind semantic shift. После 6 нейтральных ходов остается около половины средней силы. После explicit rejection тоже остается residual semantic trace: слабее, но устойчивый по знаку.

Это усиливает формулировку про persistence, но только для Qwen3-14B. Для Qwen3.5-27B остается отдельный прошлый вывод: сильная hidden geometry, но гораздо более слабый semantic readout/rejection residual.

Unembedding/logit-lens sanity check:

Contrast vector через `lm_head.weight` не дал простой картины "это просто старые labels". В top tokens есть много токенизационного/многоязычного шума (`сп`, `_SP`, `你`, `your`) и на отрицательной стороне заметен кластер `process/процесс`. Это не является доказательством смысла вектора, но полезно как negative check: эффект не выглядит как грубая проекция только на `DIRECT/VERDICT/CAUTIOUS/DISCLAIMER`.

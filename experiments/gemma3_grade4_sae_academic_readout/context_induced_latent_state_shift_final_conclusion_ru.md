# Итоговый вывод: context-induced latent-state shift в Gemma-3-12B-IT

Этот документ фиксирует текущий научный статус исследования по результатам Grade 4 анализа Gemma-3-12B-IT с SAE-res-all-small и последующего norm-controlled component-causal run. Источник фактических метрик для causal norm-control части: `C:\Users\stasv\Downloads\grade4_gemma3_12b_it_sae_res_all_small_l12_41 (1).zip`. Внутри zip использованы артефакты `red_team_input_manifest.json`, `grade4_axis_component_causal_symmetry_summary.csv`, `grade4_axis_component_causal_projection_summary.csv`, `grade4_axis_component_causal_alpha_scaling_summary.csv`, `grade4_axis_component_causal_rank_summary.csv`, `grade4_axis_component_norm_summary.csv`, `grade4_axis_component_causal_response_audit.csv` и `claim_ladder_final.csv`.

Главная доказанная гипотеза формулируется так: сильный связный target text переводит Gemma-3-12B-IT в другое измеримое внутреннее состояние во время inference, без изменения весов модели. Это не утверждение о том, что prompt влияет на финальный output. Такой эффект тривиален. Утверждение сильнее: связный target context меняет внутреннюю геометрию hidden states / residual stream, то есть меняет положение временного внутреннего состояния модели в latent space до и во время генерации ответа. В этом смысле доказанный результат является context-induced latent-state shift: сдвиг вызван контекстом, существует внутри вычисления модели и измеряется по hidden-state geometry, а не только по тексту ответа.

К этому выводу исследование пришло через последовательное отделение target-эффекта от более простых объяснений. Coherent target text сравнивался не только с обычным neutral control, но и с shuffled controls: sentence-shuffle и word-shuffle версиями того же target content. Это критично, потому что shuffled controls сохраняют значительную часть словаря, тематики, длины и локального content, но разрушают связный порядок дискурса. Если бы модель реагировала только на похожие слова, тему, длину или lexical overlap, coherent target и shuffled controls должны были бы выглядеть близко во внутренней геометрии. Они не выглядят близко. Именно это отделяет результат от банального "похожие слова активировали похожую область".

Метод, которым были получены "координаты" в latent space, был прямым hidden-state readout, а не интерпретацией финального текста. Скрипт прогонял одни и те же вопросы через разные context conditions: `target`, `neutral`, `sentence_shuffle`, `word_shuffle`, `neutral_length_matched_control`, `question_only` и другие контрольные условия. Для каждого prompt модель запускалась с `output_hidden_states=True`, после чего из каждого слоя снимался final-token hidden state. В терминах скрипта `hidden_states[0]` соответствует embedding output, а `hidden_states[1:]` соответствуют layer outputs. Для каждого условия, вопроса и слоя получался вектор в residual-stream hidden space. Поэтому исходная таблица измерений имеет смысл не "ответ модели", а "где находится внутреннее состояние модели в конце prompt-context на каждом слое".

Дальше строился не один произвольный вектор, а система осей, полученная из самих различий между условиями. Базовая идея была такой: если взять hidden state target condition и вычесть hidden state reference/neutral condition для того же вопроса и слоя, получится delta-вектор, показывающий, куда target context сдвинул внутреннее состояние относительно neutral baseline. Затем эти delta-векторы усреднялись по вопросам. Так строилась ось полного target-сдвига:

```text
x_full = mean(H_target - H_neutral)
```

Для content-like контроля строилась отдельная ось:

```text
x_content = mean(H_sentence_shuffle - H_neutral)
```

Она показывает, куда модель уходит, когда получает примерно тот же content, но без связного порядка target text. После этого строилась order difference:

```text
x_order = mean(H_target - H_sentence_shuffle)
```

И наконец из `x_order` layerwise удалялась проекция на `x_content`. Так получалась ортогонализированная order/structure component:

```text
x_order_orth = x_order - proj_x_content(x_order)
```

Это ключевой методический ход. `x_order_orth` не был придуман заранее и не был выбран вручную по красивому результату. Он был получен как остаточная component, которая остаётся после того, как из target-vs-shuffle difference удалён content-like direction. Поэтому если condition сильно проецируется на `x_order_orth`, это означает не просто "в тексте похожие слова", а "в hidden-state geometry есть сдвиг, который остаётся после отделения content signal".

Координаты в latent space получались через проекцию condition delta на найденную ось. Для каждого вопроса, слоя и condition бралась разность:

```text
delta(condition, layer, question) =
H_condition(layer, question) - H_neutral(layer, question)
```

После этого считалась координата этой delta вдоль выбранной оси:

```text
projection_fraction =
dot(delta, axis) / dot(axis, axis)
```

Это и есть численная координата состояния относительно внутреннего направления. Если `projection_fraction` положительный и большой, значит condition delta лежит в направлении этой оси. Если он около нуля, значит condition почти не движется вдоль этой оси. Если отрицательный, значит condition уходит в противоположную сторону. Дополнительно считался `direction_cosine`, то есть норм-инвариантное угловое сходство:

```text
direction_cosine =
dot(delta, axis) / (norm(delta) * norm(axis))
```

Именно поэтому результат не сводится к "длинный вектор дал большую projection". Projection показывает координату вдоль оси, а cosine проверяет, совпадает ли направление delta с направлением оси независимо от длины. В descriptive части важно, что разделение target и shuffled controls держится не только как projection/readout, но и как геометрическое разделение направлений. Это делает вывод сильнее: мы видим не просто изменение масштаба hidden state, а изменение направления в latent geometry.

Чтобы не обучить ось и проверить её на тех же самых вопросах без контроля, использовался leave-one-question-out readout. Для каждого вопроса оси пересчитывались по остальным вопросам, а затем текущий вопрос проецировался на ось, построенную без него. Это важно: если target condition всё равно стабильно проецируется на `x_order_orth`, значит readout не является простым переобучением на конкретный вопрос. Он переносится между вопросами внутри данного набора.

После получения prompt-endpoint geometry проверялось, сохраняется ли сдвиг во время генерации. Для этого модель не только обрабатывала prompt, но и генерировала ответ, а скрипт снимал hidden states во время autoregressive generation. На каждом шаге generation брались hidden states модели и сравнивались с reference prompt hidden state. Затем trajectory проецировалась на те же оси. Так появлялись generation trajectory метрики: start projection, end projection, late-minus-early projection, mean projection, direction cosine и L2 distance to reference. Это отвечает на другой вопрос: не просто "куда попал prompt endpoint", а "остаётся ли internal trajectory сдвинутой, пока модель производит ответ".

SAE readout добавлял ещё один слой проверки. Hidden states и component directions читались через sparse autoencoder features на выбранных SAE layers. Это не было основным доказательством координат в residual stream, потому что основные координаты уже получались напрямую из hidden states. SAE использовался как дополнительный sparse-feature lens: если dense residual-stream shift имеет смысл, часть различия должна проявляться как contrast в sparse features. Поэтому SAE feature contrast читался как дополнительная интерпретируемая проекция dense-сдвига, а не как единственный источник доказательства.

После descriptive geometry был сделан causal шаг. Там метод уже был другим: не только читать координаты, а вмешиваться в residual stream. Для каждой component axis (`x_order_orth` и `x_content`) скрипт создавал intervention direction и во время generation добавлял или вычитал её в выбранных layer bands:

```text
residual_state = residual_state + alpha * axis
residual_state = residual_state - alpha * axis
```

В norm-controlled run оси перед вмешательством нормировались до L2-нормы 1 по intervention band. Поэтому фактическая сила вмешательства была сопоставима между `x_order_orth` и `x_content`, а не зависела от raw length вектора. После каждого `+axis` и `-axis` run модель генерировала ответ, hidden trajectory снова снималась по слоям и шагам, и затем эта trajectory проецировалась на readout direction. Так получалась causal coordinate difference:

```text
plus_minus_projection_gap =
projection_after_plus_intervention - projection_after_minus_intervention
```

Это значение показывает не исходную координату prompt, а downstream effect intervention: насколько далеко разошлись внутренние trajectories после положительного и отрицательного сдвига residual stream. Поэтому значения вроде `+992` и `-553` являются координатными расхождениями в hidden-space readout после вмешательства. Они появились не из текста ответа и не из внешней оценки поведения, а из прямой проекции generated hidden states на найденные внутренние оси.

Именно так были "увидены" сдвиги в latent space. Сначала модель сама создавала hidden states под разными context conditions. Затем из различий между этими hidden states строились оси. Затем каждое новое condition-state или generation-state раскладывалось на координаты относительно этих осей через projection и cosine. После этого causal run проверял, меняется ли сама trajectory, если в residual stream искусственно добавить или вычесть найденную component. Поэтому результат является геометрическим в буквальном смысле: он основан на векторах hidden states, разностях этих векторов, проекциях на внутренние направления и измерении траекторий в residual-stream space.

Ключевой descriptive результат Grade 4 decomposition показывает разнос target и sentence-shuffle по разным компонентам:

```text
target on x_order_orth = 0.909026
sentence_shuffle on x_order_orth = -0.069058

sentence_shuffle on x_content = 0.849551
target on x_content = -0.010294
```

Эти числа читаются прямо. `sentence_shuffle` содержит тот же или близкий content, но уходит в `x_content`. Coherent `target` почти не загружается в `x_content`, зато сильно загружается в `x_order_orth`. Значит, измеренный сдвиг нельзя честно объяснить только content similarity. Модель не просто видит набор похожих слов. Она реагирует на связный порядок, структуру и режим обработки текста. Именно поэтому `x_order_orth` является центральной компонентой результата: она отделяет coherent target-processing mode от shuffled-content signal.

`x_order_orth` не является слабым остатком после вычитания content. Это большая часть полного target/control shift. В Grade 4 decomposition её energy fraction относительно full component составляет:

```text
middle x_order_orth_energy_fraction_of_full = 0.613503
late x_order_orth_energy_fraction_of_full = 0.564123
all x_order_orth_energy_fraction_of_full = 0.575700
```

Это означает, что order/structure/response-mode component несёт больше половины энергии полного сдвига в middle, late и all-band представлениях. Такая величина не похожа на шумовой хвост после удаления content. Это крупная геометрическая компонента, отделимая от content-like направления. Поэтому корректная интерпретация descriptive результата такая: связный target context создаёт в модели не только semantic/content activation, но и отдельный структурный latent-state shift, связанный с порядком, дискурсивной связностью и режимом ответа.

После descriptive доказательства следующий вопрос был причинным. Нужно было понять, является ли найденная component direction только координатой, которая хорошо читает уже возникший сдвиг, или вмешательство по этой component direction само двигает generation trajectory. Поэтому был проведён component-causal run по осям `x_order_orth` и `x_content`. Но первый raw-alpha causal результат имел важный confound: вмешательство было устроено как `residual_state = residual_state + alpha * vector`. При такой формуле одинаковый `alpha` не означает одинаковую силу вмешательства, если raw L2-нормы векторов разные. Длинный вектор получает физически более сильное вмешательство просто из-за своей длины.

Новый norm-controlled causal run был нужен именно для закрытия этого confound. В новом zip raw-нормы компонент подтверждают проблему:

```text
middle x_content raw norm = 14518.902068
middle x_order_orth raw norm = 8058.432071

late x_content raw norm = 29315.891582
late x_order_orth raw norm = 14729.571563
```

`x_content` в raw representation был примерно в 1.8-2.0 раза длиннее, чем `x_order_orth`. Поэтому в raw-alpha setting `x_content` мог выигрывать не потому, что он причинно более важен, а потому что в residual stream добавлялся более длинный вектор. Для вопроса "какая component direction причинно сильнее двигает generation trajectory" такое сравнение нечистое. Оно смешивает направление оси и физическую энергию вмешательства.

Norm-controlled run сделал ровно то, что нужно для честного directional comparison: обе оси, `x_order_orth` и `x_content`, нормировались до L2-нормы 1 по выбранному intervention band. После этого применялись одинаковые `alpha` values `0.25`, `0.50`, `0.75` на middle и late layer bands, с base conditions `neutral` и `target`. Manifest прогона фиксирует:

```text
model_id = google/gemma-3-12b-it
run_label = grade4_gemma3_12b_it_sae_res_all_small_l12_41_causal_xorder_normctl

GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_ENABLED = True
GRADE4_COMPONENT_CAUSAL_NORM_CONTROL_MODE = band_l2
GRADE4_COMPONENT_CAUSAL_READOUT_USES_NORMED_AXIS = True

CAUSAL_INTERVENTIONS_ENABLED = False
BEHAVIORAL_CONTROL_AXIS_ENABLED = False
```

Это был именно component-causal norm-control test. Он не был full behavioral steering run и не тестировал финальное поведение как главный критерий. Он проверял внутреннюю generation trajectory после controlled intervention в residual stream. Технически norm-control сработал: по symmetry/projection summary видно, что `mean_intervention_axis_band_norm = 1.0`, а `mean_effective_intervention_l2 = alpha_abs`. То есть при `alpha = 0.25` обе оси вмешивались с L2-силой около `0.25`, при `alpha = 0.50` с силой около `0.50`, при `alpha = 0.75` с силой около `0.75`. После этого `x_content` уже не мог выигрывать только потому, что его raw vector длиннее.

Главная величина causal readout в этом прогоне — `plus_minus_projection_gap`:

```text
plus_minus_projection_gap =
projection_after_plus_intervention - projection_after_minus_intervention
```

Эта величина не является процентом, accuracy, вероятностью или оценкой качества ответа. Это разность координат в residual-stream readout space. Мы берём hidden states модели во время generation trajectory, проецируем их на выбранную внутреннюю ось и сравниваем два вмешательства: `+axis` и `-axis`. Если gap равен `+992`, это значит, что trajectory после `+axis` оказалась примерно на 992 projection-units выше вдоль этой внутренней координаты, чем trajectory после `-axis`. Если gap равен `-1211`, это значит, что trajectories тоже сильно разошлись, но направление оказалось обратным: `-axis` дал более высокую координату, или dynamics развернулась несимметрично.

Поэтому большие числа в этом прогоне — это большие hidden-space readout shifts. Они означают, что маленькое controlled intervention в residual stream может дать большое downstream-расхождение внутренней generation trajectory. Это именно тот тип сдвига в пространстве, который искался: не изменение весов, не изменение "личности" модели, а временное изменение траектории hidden states во время inference. Однако большая амплитуда сама по себе не равна управляемости. Для stable causal steering нужна не только величина, но и стабильное направление, симметрия `+/-`, воспроизводимость по base conditions и нормальная dose-response зависимость от alpha.

Фактический aggregate result norm-controlled causal run:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate = 0.527778

x_content mean causal gap = -125.128343
x_content positive rate = 0.472222

all readouts: x_order_orth beats x_content = 0.416667
matching readouts only: x_order_orth beats x_content = 0.500000
```

После честного unit-L2 сравнения `x_order_orth` не стал устойчивым победителем над `x_content`. Он имеет slightly higher positive rate и менее отрицательный mean gap, но pairwise dominance отсутствует: по всем readouts он выигрывает у `x_content` только в `0.416667` случаев, а по matching readouts, где intervention band совпадает с readout band, результат ровно `0.500000`. Это фиксирует границу causal claim: `x_order_orth` не доказан как dominant causal component в unit-L2 setting.

Главная структура результата находится в разделении по base condition:

```text
neutral: x_order_orth beats x_content = 0.666667
neutral mean order_minus_content_gap = +354.870122

target: x_order_orth beats x_content = 0.166667
target mean order_minus_content_gap = -236.496475
```

Это важнее общего среднего. В neutral condition `x_order_orth` явно ведёт себя сильнее: добавление `x_order_orth` в нейтральное состояние чаще и сильнее двигает generation trajectory, чем добавление `x_content`. Mean advantage `+354.870122` означает, что в среднем по paired cells `x_order_orth` даёт значительно более высокий plus/minus projection gap, чем `x_content`. Это causal signal, а не пустой readout. Ось, которая была найдена descriptive decomposition, действительно способна участвовать в движении внутренней trajectory.

В target condition картина обратная. Там `x_order_orth` выигрывает у `x_content` только в `0.166667` paired cells, а mean `order_minus_content_gap = -236.496475`. Это означает, что вычитание/симметричное вмешательство по `x_order_orth` из уже target-conditioned состояния не работает как зеркальный выключатель. На нормальном языке: `x_order_orth` лучше работает как injection direction из neutral, чем как стабильная bidirectional ручка, которая одинаково чисто вводит модель в target-like state и выводит её из него. Это не разрушает descriptive доказательство latent-state shift. Это показывает, что causal dynamics не является простой линейной кнопкой.

Rank summary на `alpha_abs = 0.75` даёт конкретные примеры больших сдвигов:

```text
neutral late x_order_orth gap = +992.518931
neutral late x_content gap = +356.819982

neutral middle x_order_orth gap = +274.611926
neutral middle x_content gap = +52.812108

target late x_content gap = -51.775190
target late x_order_orth gap = -553.394467

target middle x_content gap = +459.055941
target middle x_order_orth gap = -0.605217
```

Эти строки показывают механику результата вживую. В neutral condition `x_order_orth` даёт крупный internal trajectory shift: особенно в late band gap `+992.518931`, что существенно выше `x_content` gap `+356.819982` при той же effective intervention L2 около `0.75`. В middle band neutral картина такая же: `x_order_orth` gap `+274.611926` против `x_content` gap `+52.812108`. Но в target condition зеркальность ломается. В late target `x_order_orth` уходит в `-553.394467`, а в middle target почти не даёт положительного эффекта (`-0.605217`), тогда как `x_content` в middle target имеет `+459.055941`. Значит, intervention по `x_order_orth` действительно двигает внутреннее состояние, но не даёт стабильной bidirectional control symmetry.

Alpha scaling подтверждает это ограничение:

```text
x_order_orth signed alpha slope mean = -23.426489
x_order_orth positive slope rate = 0.250000

x_content signed alpha slope mean = -121.248341
x_content positive slope rate = 0.416667
```

Если бы `x_order_orth` был чистой steering axis, увеличение `alpha` должно было бы давать устойчивое усиление эффекта в ожидаемом направлении. Этого нет. Projection range по alpha большой, trajectories чувствительны, но signed slope не демонстрирует стабильного положительного dose-response. Это означает, что causal sensitivity есть, а stable steering control в unit-L2 setting не доказан.

Итоговый научный вывод должен быть зафиксирован жёстко. Мы доказали context-induced latent-state shift: сильный coherent target text переводит Gemma-3-12B-IT в другое измеримое внутреннее состояние. Этот сдвиг не сводится к content/lexical overlap, потому что shuffled-content controls расходятся с target по разным компонентам: sentence-shuffle уходит в `x_content`, coherent target уходит в `x_order_orth`. Мы показали, что `x_order_orth` является крупной order/structure/response-mode component, а не слабым остатком. Мы также показали, что component directions не являются полностью пассивными описательными координатами: controlled intervention по ним вызывает измеримые generation-trajectory shifts в residual-stream space. Однако norm-controlled causal run не доказывает, что `x_order_orth` является стабильной bidirectional steering axis. Текущий статус результата: descriptive proof strong; causal involvement supported; stable causal control not proven.

Для всего исследования это означает следующее. Главный результат не сужается и не отменяется causal norm-control прогоном. Главный доказанный результат остаётся внутренним latent-state shift, вызванный связным target context и отделённый от content/shuffle controls. Norm-controlled causal run отвечает на следующий, более строгий вопрос: является ли найденная `x_order_orth` простой causal control handle. Ответ по unit-L2 run: нет, не как простая стабильная bidirectional ручка. Механизм реальный, но асимметричный. Он больше похож на нелинейную trajectory dynamics, где вход из neutral в target-like direction и выход из target-conditioned state не обязаны быть зеркальными операциями.

Это важное уточнение, а не откат результата. Исследование перешло от "есть ли измеримый latent shift" к "какова механика найденной component direction". Ответ сейчас такой: `x_order_orth` сильна как readout/separation component, имеет causal signal, особенно при neutral injection, но её causal steering не закрыт на уровне стабильного управления. Значит, следующий эксперимент должен проверять не сам факт latent-state shift — он уже установлен descriptive evidence, — а масштаб и форму causal intervention.

Следующий шаг должен быть `natural-scale norm-controlled causal run`. Unit-L2 intervention был честным по энергии, но очень маленьким относительно natural raw norms: для `x_order_orth` raw norm составляет примерно `8058` в middle band и `14730` в late band, тогда как effective intervention L2 в normctl run был только `0.25`, `0.50`, `0.75`. Такой прогон честно сравнил направления, но мог быть слишком слабым относительно естественного масштаба latent shift. Следующий run должен сохранить fairness сравнения, то есть не дать `x_content` преимущества raw norm, но вернуть natural-scale magnitude через общий natural band norm. Это позволит проверить, является ли нестабильность unit-L2 результата следствием слабого масштаба intervention или настоящей границей `x_order_orth` как steering direction.

Финальная формулировка текущего статуса исследования:

```text
Доказано: coherent target text вызывает context-induced latent-state shift в
Gemma-3-12B-IT. Этот shift измеряется во внутренней геометрии hidden states /
residual stream, отделяется от shuffled-content controls и содержит крупную
order/structure component x_order_orth.

Поддержано causal evidence: intervention по найденным component directions
изменяет generation trajectory, то есть эти directions не являются чисто
пассивными координатами.

Не доказано: x_order_orth как стабильная bidirectional steering axis или
полная ручка управления поведением модели.

Следующий необходимый тест: natural-scale norm-controlled component-causal run.
```

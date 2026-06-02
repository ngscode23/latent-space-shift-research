# Norm-Controlled Causal Run Conclusion

## Вывод на русском

Этот прогон не проверял заново, существует ли descriptive latent shift. Этот
вопрос уже был проверен предыдущим Grade 4 анализом: coherent target text
отделяется от sentence-shuffle/content controls во внутренней геометрии модели,
и это разделение видно не только в projection, но и в decomposition,
cosine/readout и SAE feature contrast.

Этот новый прогон проверял другой, более строгий вопрос:

```text
если x_order_orth и x_content вмешивать в residual stream с одинаковой L2-силой,
какая component direction причинно сильнее двигает generation trajectory?
```

То есть это был causal direction test, а не descriptive geometry test.

Причина прогона была простая: в предыдущем raw-alpha causal run `x_content`
мог получить преимущество только потому, что его raw vector norm намного больше,
чем у `x_order_orth`. Там intervention была вида:

```text
residual = residual + alpha * vector
```

При такой формуле одинаковый `alpha` не означает одинаковую силу вмешательства,
если длины векторов разные. Поэтому мы сделали norm-control: обе component
directions нормировались до L2-нормы 1 по выбранному intervention band.

Norm-control сработал:

```text
mean_intervention_axis_band_norm = 1.0
mean_effective_intervention_l2 = alpha_abs
```

Это значит, что старый raw-norm confound был закрыт. `x_content` больше не имел
преимущества просто из-за большей длины raw vector.

Главный результат:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate   = 0.527778

x_content mean causal gap    = -125.128343
x_content positive rate      = 0.472222
```

В paired comparison:

```text
all readouts:
  x_order_orth beats x_content = 0.416667

matching readouts only:
  x_order_orth beats x_content = 0.500000
```

Это не победа `x_order_orth`. Это результат уровня нестабильного/смешанного
сигнала. После честного unit-L2 сравнения `x_order_orth` не показал устойчивого
causal dominance над `x_content`.

Самая важная структура результата находится не в общем среднем, а в разбиении
по base condition:

```text
neutral:
  x_order_orth beats x_content = 0.666667
  mean order_minus_content_gap = +354.870122

target:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -236.496475
```

Это значит, что `x_order_orth` лучше работает как injection direction:
`neutral + x_order_orth` иногда двигает trajectory в target-like сторону.
Но обратная операция `target - x_order_orth` не даёт устойчивого зеркального
эффекта. Поэтому bidirectional causal symmetry пока не доказана.

Alpha scaling тоже слабый:

```text
x_order_orth positive slope rate = 0.250000
x_content positive slope rate    = 0.416667
```

То есть при росте alpha эффект не растёт стабильно. Это снижает силу causal
claim.

Итоговая интерпретация:

```text
Unit-L2 norm-controlled run убрал raw-norm confound, но не доказал, что
x_order_orth является dominant causal steering component. Он показывает
слабый/асимметричный causal signal, особенно для neutral + x_order_orth
injection, но не показывает чистой bidirectional symmetry и dose-response.
```

Что это значит для исследования:

```text
1. Descriptive Grade 4 result остается сильным.
2. x_order_orth остается важной separable latent component.
3. Raw-alpha causal claim был ограничен norm confound.
4. Unit-L2 normctl run закрывает norm confound, но оказывается underpowered
   для natural-scale steering.
5. Causal dominance x_order_orth над x_content пока не доказана.
```

Ключевой технический момент: unit-L2 intervention честно сравнивает directions,
но делает intervention очень маленькой относительно реального масштаба latent
shift. Natural component norms были тысячами:

```text
middle x_order_orth raw norm ≈ 8058
late x_order_orth raw norm   ≈ 14730
```

А unit-L2 intervention использовал effective L2:

```text
0.25, 0.50, 0.75
```

Это на несколько порядков меньше natural component scale. Поэтому этот прогон
может быть честным как direction-comparison, но слишком слабым как проверка
natural-scale causal control.

Следующий правильный эксперимент:

```text
norm-controlled natural-scale causal run
```

Технически:

```text
unit_x_order_orth = x_order_orth / norm(x_order_orth over band)
unit_x_content    = x_content / norm(x_content over band)

shared_band_norm = min(norm(x_order_orth over band), norm(x_content over band))

intervention = alpha * shared_band_norm * unit_axis
```

Это сохранит честное equal-energy сравнение, но вернёт intervention в масштаб,
сопоставимый с реальным latent shift.

Самый короткий вывод:

```text
Этот прогон не опроверг descriptive latent-state shift. Он показал, что
unit-L2 causal steering через x_order_orth пока неустойчив. Следующий шаг —
проверить те же directions при equal-energy, но natural-scale intervention.
```

## English Conclusion

This run did not retest whether the descriptive latent shift exists. That was
already tested in the earlier Grade 4 analysis: coherent target text separates
from sentence-shuffled/content controls in the model's internal geometry, and
that separation appears across projection, decomposition, cosine/readout, and
SAE feature contrast.

This new run tested a narrower and stricter causal question:

```text
If x_order_orth and x_content are injected into the residual stream with equal
L2 intervention strength, which component direction causally moves the
generation trajectory more strongly?
```

So this was a causal direction test, not a descriptive geometry test.

The reason for this run was that the previous raw-alpha causal run had a norm
confound. `x_content` had a much larger raw vector norm than `x_order_orth`.
With an intervention of the form:

```text
residual = residual + alpha * vector
```

the same alpha does not mean the same intervention strength if the vectors have
different norms. The norm-controlled run fixed this by normalizing both
component directions to unit L2 norm over the selected intervention band.

The norm-control worked:

```text
mean_intervention_axis_band_norm = 1.0
mean_effective_intervention_l2 = alpha_abs
```

This means the raw-norm advantage was removed. `x_content` no longer had a
larger perturbation just because its raw vector was longer.

Main result:

```text
x_order_orth mean causal gap = -65.941520
x_order_orth positive rate   = 0.527778

x_content mean causal gap    = -125.128343
x_content positive rate      = 0.472222
```

Paired comparison:

```text
all readouts:
  x_order_orth beats x_content = 0.416667

matching readouts only:
  x_order_orth beats x_content = 0.500000
```

This is not a win for `x_order_orth`. It is a mixed and unstable signal. Under
fair unit-L2 comparison, `x_order_orth` does not show stable causal dominance
over `x_content`.

The most important structure is in the base-condition split:

```text
neutral:
  x_order_orth beats x_content = 0.666667
  mean order_minus_content_gap = +354.870122

target:
  x_order_orth beats x_content = 0.166667
  mean order_minus_content_gap = -236.496475
```

This means `x_order_orth` works better as an injection direction:
`neutral + x_order_orth` sometimes moves the trajectory in the target-like
direction. But the reverse operation, `target - x_order_orth`, does not show a
stable mirror effect. Therefore bidirectional causal symmetry is not yet
established.

Alpha scaling is also weak:

```text
x_order_orth positive slope rate = 0.250000
x_content positive slope rate    = 0.416667
```

The effect does not grow cleanly with alpha, which weakens the causal claim.

Correct interpretation:

```text
The unit-L2 norm-controlled run removed the raw-norm confound, but it did not
prove that x_order_orth is the dominant causal steering component. It shows a
weak/asymmetric causal signal, especially for neutral + x_order_orth injection,
but not clean bidirectional symmetry or dose-response.
```

What this means for the research:

```text
1. The descriptive Grade 4 result remains strong.
2. x_order_orth remains an important separable latent component.
3. The raw-alpha causal claim was limited by a norm confound.
4. The unit-L2 normctl run removes that confound, but is likely underpowered
   for natural-scale steering.
5. Causal dominance of x_order_orth over x_content is not yet established.
```

The key technical point is that unit-L2 intervention fairly compares directions,
but makes the perturbation extremely small relative to the natural latent-shift
scale. The natural component norms were in the thousands:

```text
middle x_order_orth raw norm ≈ 8058
late x_order_orth raw norm   ≈ 14730
```

The unit-L2 run used effective intervention strengths of only:

```text
0.25, 0.50, 0.75
```

That is several orders of magnitude smaller than the natural component scale.
So this run is fair as a direction comparison, but may be too weak as a test of
natural-scale causal control.

The next correct experiment is:

```text
norm-controlled natural-scale causal run
```

Technically:

```text
unit_x_order_orth = x_order_orth / norm(x_order_orth over band)
unit_x_content    = x_content / norm(x_content over band)

shared_band_norm = min(norm(x_order_orth over band), norm(x_content over band))

intervention = alpha * shared_band_norm * unit_axis
```

This keeps the comparison fair while making the intervention comparable to the
natural latent-shift scale.

Shortest conclusion:

```text
This run does not refute the descriptive latent-state shift. It shows that
unit-L2 causal steering through x_order_orth is not yet stable. The next step is
to test the same directions with equal-energy but natural-scale intervention.
```

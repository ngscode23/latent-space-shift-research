# Causal Mediation v1 Report

Model: `Qwen/Qwen3-14B`
Texts per kind: `9`
Selected hidden indices: `[40, 32]`

## Interpretation Rule

`toward_expected_fraction` is the fraction of the natural target-control gap recovered by `control + vector` or reduced by `target - vector`.

Strong internal success starts around `>= 0.30` with bootstrap lower bound above zero, and target-control vector should beat random/shuffled controls.

## Best Bootstrap Rows

| readout_type | hidden_index | vector_kind | alpha_magnitude | intervention_kind | observed | ci_low | ci_high | n_units | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_action | 32 | target_control | 1.0000 | control_plus | 0.8898 | 0.4768 | 1.2191 | 9 | 36 |
| agent_action | 32 | target_control | 0.5000 | control_plus | 0.3939 | 0.1949 | 0.5505 | 9 | 36 |
| agent_action | 32 | wrong_layer | 1.0000 | control_plus | 0.2989 | -0.0735 | 0.5708 | 9 | 36 |
| agent_action | 32 | wrong_layer | 0.5000 | control_plus | 0.0329 | -0.2322 | 0.2098 | 9 | 36 |
| agent_action | 32 | random_same_norm | 1.0000 | control_plus | 0.0129 | -0.2579 | 0.1830 | 9 | 36 |
| agent_action | 40 | random_same_norm | 1.0000 | control_plus | 0.0066 | -0.0079 | 0.0172 | 9 | 36 |
| agent_action | 40 | wrong_layer | 1.0000 | control_plus | 0.0033 | -0.0143 | 0.0170 | 9 | 36 |
| agent_action | 40 | random_same_norm | 0.5000 | control_plus | -0.0030 | -0.0081 | 0.0023 | 9 | 36 |
| agent_action | 40 | wrong_layer | 0.5000 | control_plus | -0.0075 | -0.0242 | 0.0029 | 9 | 36 |
| agent_action | 40 | shuffled_label | 0.5000 | control_plus | -0.0276 | -0.0427 | -0.0161 | 9 | 36 |
| agent_action | 40 | target_control | 0.5000 | control_plus | -0.0340 | -0.0529 | -0.0195 | 9 | 36 |
| agent_action | 40 | shuffled_label | 1.0000 | control_plus | -0.0398 | -0.0718 | -0.0159 | 9 | 36 |
| agent_action | 32 | random_same_norm | 0.5000 | control_plus | -0.0483 | -0.1749 | 0.0253 | 9 | 36 |
| agent_action | 40 | target_control | 1.0000 | control_plus | -0.0640 | -0.1223 | -0.0236 | 9 | 36 |
| agent_action | 32 | shuffled_label | 0.5000 | control_plus | -0.0764 | -0.1383 | 0.0004 | 9 | 36 |
| agent_action | 32 | shuffled_label | 1.0000 | control_plus | -0.1855 | -0.3719 | -0.0183 | 9 | 36 |
| agent_action | 32 | target_control | 1.0000 | target_minus | 0.2142 | 0.1046 | 0.3194 | 9 | 36 |
| agent_action | 32 | target_control | 0.5000 | target_minus | 0.0543 | -0.0821 | 0.1532 | 9 | 36 |
| agent_action | 40 | random_same_norm | 0.5000 | target_minus | -0.0044 | -0.0081 | -0.0008 | 9 | 36 |
| agent_action | 40 | wrong_layer | 0.5000 | target_minus | -0.0119 | -0.0262 | -0.0023 | 9 | 36 |
| agent_action | 40 | random_same_norm | 1.0000 | target_minus | -0.0190 | -0.0274 | -0.0120 | 9 | 36 |
| agent_action | 32 | wrong_layer | 0.5000 | target_minus | -0.0202 | -0.1843 | 0.0928 | 9 | 36 |
| agent_action | 40 | wrong_layer | 1.0000 | target_minus | -0.0221 | -0.0395 | -0.0085 | 9 | 36 |
| agent_action | 40 | shuffled_label | 0.5000 | target_minus | -0.0230 | -0.0370 | -0.0114 | 9 | 36 |
| agent_action | 40 | target_control | 0.5000 | target_minus | -0.0347 | -0.0620 | -0.0132 | 9 | 36 |
| agent_action | 32 | wrong_layer | 1.0000 | target_minus | -0.0465 | -0.3047 | 0.1391 | 9 | 36 |
| agent_action | 40 | shuffled_label | 1.0000 | target_minus | -0.0472 | -0.0732 | -0.0252 | 9 | 36 |
| agent_action | 40 | target_control | 1.0000 | target_minus | -0.0669 | -0.1141 | -0.0314 | 9 | 36 |
| agent_action | 32 | random_same_norm | 0.5000 | target_minus | -0.0762 | -0.1600 | -0.0188 | 9 | 36 |
| agent_action | 32 | random_same_norm | 1.0000 | target_minus | -0.1780 | -0.3070 | -0.0805 | 9 | 36 |
| agent_action | 32 | shuffled_label | 0.5000 | target_minus | -0.3618 | -0.4761 | -0.2503 | 9 | 36 |
| agent_action | 32 | shuffled_label | 1.0000 | target_minus | -0.7788 | -1.0168 | -0.5593 | 9 | 36 |
| blind_semantic | 32 | target_control | 1.0000 | control_plus | 0.2911 | 0.2160 | 0.3847 | 9 | 72 |
| blind_semantic | 32 | shuffled_label | 1.0000 | control_plus | 0.1285 | 0.0965 | 0.1661 | 9 | 72 |
| blind_semantic | 32 | target_control | 0.5000 | control_plus | 0.1190 | 0.0861 | 0.1604 | 9 | 72 |
| blind_semantic | 32 | shuffled_label | 0.5000 | control_plus | 0.0595 | 0.0467 | 0.0736 | 9 | 72 |
| blind_semantic | 32 | wrong_layer | 1.0000 | control_plus | 0.0561 | 0.0445 | 0.0711 | 9 | 72 |
| blind_semantic | 32 | wrong_layer | 0.5000 | control_plus | 0.0124 | 0.0088 | 0.0170 | 9 | 72 |
| blind_semantic | 40 | random_same_norm | 1.0000 | control_plus | 0.0065 | 0.0048 | 0.0084 | 9 | 72 |
| blind_semantic | 40 | wrong_layer | 1.0000 | control_plus | 0.0060 | 0.0042 | 0.0081 | 9 | 72 |

## Target-Control Success Rows

| readout_type | hidden_index | module_layer | vector_kind | alpha_magnitude | intervention_kind | observed | ci_low | ci_high | n_units | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_action | 32 | 31 | target_control | 0.5000 | control_plus | 0.3939 | 0.1949 | 0.5505 | 9 | 36 |
| agent_action | 32 | 31 | target_control | 1.0000 | control_plus | 0.8898 | 0.4768 | 1.2191 | 9 | 36 |

## Notes

- If target_control does not beat random/shuffled, do not claim vector-level mediation.
- If semantic succeeds but agent_action fails, claim semantic mediation only.
- If both succeed, upgrade the causal chain to partial mediation.
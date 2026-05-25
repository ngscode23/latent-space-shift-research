# Causal Mediation v1 Report

Model: `mistralai/Ministral-3-14B-Instruct-2512-BF16`
Texts per kind: `9`
Selected hidden indices: `[14, 13, 40]`

## Interpretation Rule

`toward_expected_fraction` is the fraction of the natural target-control gap recovered by `control + vector` or reduced by `target - vector`.

Strong internal success starts around `>= 0.30` with bootstrap lower bound above zero, and target-control vector should beat random/shuffled controls.

## Best Bootstrap Rows

| readout_type | hidden_index | vector_kind | alpha_magnitude | intervention_kind | observed | ci_low | ci_high | n_units | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_action | 40 | random_same_norm | 1.0000 | control_plus | 0.1633 | 0.1334 | 0.1929 | 9 | 36 |
| agent_action | 40 | shuffled_label | 1.0000 | control_plus | 0.0952 | 0.0745 | 0.1171 | 9 | 36 |
| agent_action | 40 | target_control | 1.0000 | control_plus | 0.0745 | 0.0451 | 0.1071 | 9 | 36 |
| agent_action | 40 | random_same_norm | 0.5000 | control_plus | 0.0622 | 0.0505 | 0.0752 | 9 | 36 |
| agent_action | 40 | wrong_layer | 1.0000 | control_plus | 0.0552 | 0.0253 | 0.0868 | 9 | 36 |
| agent_action | 14 | random_same_norm | 1.0000 | control_plus | 0.0320 | 0.0237 | 0.0402 | 9 | 36 |
| agent_action | 13 | shuffled_label | 1.0000 | control_plus | 0.0289 | 0.0192 | 0.0373 | 9 | 36 |
| agent_action | 13 | shuffled_label | 0.5000 | control_plus | 0.0226 | 0.0186 | 0.0267 | 9 | 36 |
| agent_action | 40 | shuffled_label | 0.5000 | control_plus | 0.0204 | 0.0136 | 0.0277 | 9 | 36 |
| agent_action | 40 | target_control | 0.5000 | control_plus | 0.0117 | -0.0026 | 0.0257 | 9 | 36 |
| agent_action | 14 | random_same_norm | 0.5000 | control_plus | 0.0114 | 0.0083 | 0.0143 | 9 | 36 |
| agent_action | 14 | wrong_layer | 1.0000 | control_plus | 0.0027 | -0.0057 | 0.0110 | 9 | 36 |
| agent_action | 13 | random_same_norm | 0.5000 | control_plus | -0.0012 | -0.0053 | 0.0024 | 9 | 36 |
| agent_action | 40 | wrong_layer | 0.5000 | control_plus | -0.0018 | -0.0179 | 0.0128 | 9 | 36 |
| agent_action | 13 | target_control | 1.0000 | control_plus | -0.0051 | -0.0123 | 0.0008 | 9 | 36 |
| agent_action | 13 | wrong_layer | 1.0000 | control_plus | -0.0064 | -0.0127 | -0.0003 | 9 | 36 |
| agent_action | 13 | target_control | 0.5000 | control_plus | -0.0065 | -0.0096 | -0.0035 | 9 | 36 |
| agent_action | 13 | random_same_norm | 1.0000 | control_plus | -0.0094 | -0.0164 | -0.0033 | 9 | 36 |
| agent_action | 13 | wrong_layer | 0.5000 | control_plus | -0.0121 | -0.0172 | -0.0078 | 9 | 36 |
| agent_action | 14 | shuffled_label | 1.0000 | control_plus | -0.0141 | -0.0224 | -0.0065 | 9 | 36 |
| agent_action | 14 | target_control | 1.0000 | control_plus | -0.0152 | -0.0230 | -0.0086 | 9 | 36 |
| agent_action | 14 | wrong_layer | 0.5000 | control_plus | -0.0166 | -0.0238 | -0.0097 | 9 | 36 |
| agent_action | 14 | shuffled_label | 0.5000 | control_plus | -0.0203 | -0.0248 | -0.0160 | 9 | 36 |
| agent_action | 14 | target_control | 0.5000 | control_plus | -0.0204 | -0.0264 | -0.0153 | 9 | 36 |
| agent_action | 40 | random_same_norm | 1.0000 | target_minus | 0.1186 | 0.0937 | 0.1415 | 9 | 36 |
| agent_action | 13 | shuffled_label | 1.0000 | target_minus | 0.1100 | 0.0872 | 0.1346 | 9 | 36 |
| agent_action | 40 | shuffled_label | 1.0000 | target_minus | 0.0728 | 0.0466 | 0.1013 | 9 | 36 |
| agent_action | 14 | random_same_norm | 1.0000 | target_minus | 0.0687 | 0.0585 | 0.0798 | 9 | 36 |
| agent_action | 14 | wrong_layer | 1.0000 | target_minus | 0.0669 | 0.0487 | 0.0885 | 9 | 36 |
| agent_action | 13 | shuffled_label | 0.5000 | target_minus | 0.0536 | 0.0427 | 0.0645 | 9 | 36 |
| agent_action | 40 | random_same_norm | 0.5000 | target_minus | 0.0481 | 0.0379 | 0.0584 | 9 | 36 |
| agent_action | 14 | random_same_norm | 0.5000 | target_minus | 0.0347 | 0.0285 | 0.0406 | 9 | 36 |
| agent_action | 13 | wrong_layer | 1.0000 | target_minus | 0.0259 | 0.0099 | 0.0451 | 9 | 36 |
| agent_action | 40 | shuffled_label | 0.5000 | target_minus | 0.0231 | 0.0119 | 0.0351 | 9 | 36 |
| agent_action | 14 | wrong_layer | 0.5000 | target_minus | 0.0195 | 0.0108 | 0.0302 | 9 | 36 |
| agent_action | 40 | wrong_layer | 1.0000 | target_minus | 0.0141 | -0.0245 | 0.0510 | 9 | 36 |
| agent_action | 40 | target_control | 1.0000 | target_minus | 0.0103 | -0.0241 | 0.0429 | 9 | 36 |
| agent_action | 13 | wrong_layer | 0.5000 | target_minus | 0.0028 | -0.0058 | 0.0129 | 9 | 36 |
| agent_action | 14 | shuffled_label | 1.0000 | target_minus | -0.0050 | -0.0141 | 0.0031 | 9 | 36 |
| agent_action | 14 | target_control | 1.0000 | target_minus | -0.0072 | -0.0199 | 0.0049 | 9 | 36 |

## Target-Control Success Rows

_empty_

## Notes

- If target_control does not beat random/shuffled, do not claim vector-level mediation.
- If semantic succeeds but agent_action fails, claim semantic mediation only.
- If both succeed, upgrade the causal chain to partial mediation.
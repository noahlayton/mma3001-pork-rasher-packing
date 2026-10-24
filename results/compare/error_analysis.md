# Error analysis and statistical comparison (test split)

## Paired bootstrap: metric(B) - metric(A), 95% CI

For recall, positive = B catches more defective trays. For false_reject_rate, negative = B wastes fewer good trays. An interval containing 0 means the difference is not statistically established on this test set. `p_le_0` is the share of resamples with diff <= 0 (for recall, a one-sided p-value for 'B is better').

| A           | B       | metric            |    diff |   lower |   upper |   p_le_0 |
|:------------|:--------|:------------------|--------:|--------:|--------:|---------:|
| baseline    | y8n_640 | recall            |  0.0588 | -0.058  |  0.1765 |   0.1788 |
| baseline    | y8n_640 | false_reject_rate | -0.25   | -0.5    |  0      |   1      |
| y8n_640     | y8s_640 | recall            |  0.0147 | -0.0597 |  0.0909 |   0.42   |
| y8n_640     | y8s_640 | false_reject_rate |  0      |  0      |  0      |   1      |
| y8n_640     | y8n_960 | recall            |  0.0735 |  0      |  0.1515 |   0.033  |
| y8n_640     | y8n_960 | false_reject_rate |  0      |  0      |  0      |   1      |
| y8n_320     | y8n_640 | recall            |  0.0294 | -0.0455 |  0.1111 |   0.2964 |
| y8n_320     | y8n_640 | false_reject_rate |  0      |  0      |  0      |   1      |
| y8n_640_f50 | y8n_640 | recall            | -0.0147 | -0.0694 |  0.0312 |   0.816  |
| y8n_640_f50 | y8n_640 | false_reject_rate | -0.1667 | -0.4167 |  0      |   1      |
| y8n_640_f25 | y8n_640 | recall            |  0.0588 | -0.0299 |  0.1528 |   0.1194 |
| y8n_640_f25 | y8n_640 | false_reject_rate | -0.0833 | -0.2727 |  0      |   1      |

## Good trays rejected by each method

- **y8n_640**: 1 — frame_with_red_7471
- **y8s_640**: 1 — frame_with_red_7471
- **y8n_320**: 1 — frame_with_red_7471
- **y8n_960**: 1 — frame_with_red_7471
- **y8n_640_f50**: 3 — frame_with_red_3482, frame_with_red_62, frame_with_red_7471
- **y8n_640_f25**: 2 — frame_with_red_2248, frame_with_red_7471
- **baseline**: 4 — frame_with_red_2248, frame_with_red_4088, frame_with_red_595, frame_with_red_7471

Rejected by **every** method: frame_with_red_7471_jpg.rf.d9733c9a3c8e75ec75d4c6c9f744a199.jpg. A tray that every independent method flags is a candidate labelling error.

## Defective trays missed by y8n_640

| class | test trays containing it | caught | missed |
|---|---|---|---|
| loose-meat | 2 | 1 | 1 |
| packaging-error | 16 | 15 | 1 |
| twisted-meat | 4 | 0 | 4 |
| unsealed | 49 | 45 | 4 |

Total missed: 10 of 68 defective trays.

## AP50 on the reliably-evaluated classes only

| model | unsealed | packaging-error | mean |
|---|---|---|---|
| y8n_640 | 0.828 | 0.520 | 0.674 |
| y8s_640 | 0.893 | 0.450 | 0.672 |
| y8n_320 | 0.852 | 0.449 | 0.650 |
| y8n_960 | 0.838 | 0.440 | 0.639 |
| y8n_640_f50 | 0.836 | 0.518 | 0.677 |
| y8n_640_f25 | 0.803 | 0.278 | 0.540 |

## Tray recall at fixed thresholds (test)

| model | 0.15 | 0.20 | 0.25 | 0.30 | chosen on valid |
|---|---|---|---|---|---|
| y8n_640 | 0.897 | 0.882 | 0.868 | 0.853 | 0.3 |
| y8s_640 | 0.956 | 0.956 | 0.941 | 0.912 | 0.4 |
| y8n_320 | 0.941 | 0.926 | 0.897 | 0.824 | 0.3 |
| y8n_960 | 0.926 | 0.897 | 0.868 | 0.765 | 0.15 |
| y8n_640_f50 | 0.926 | 0.912 | 0.912 | 0.868 | 0.3 |
| y8n_640_f25 | 0.838 | 0.809 | 0.794 | 0.750 | 0.25 |

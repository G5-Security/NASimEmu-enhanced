# Eval results — Pen-DHRL-TD (final checkpoint) vs. Pen-DHRL baseline

Pen-DHRL-TD = `model.pt` (train_step 20000, final) from LLM-distilled run `b33undf9`.
Pen-DHRL baseline = same architecture, same parameter count (212k), trained without
`--llm_distill`. The baseline was trained twice independently; per scenario the
stronger of the two runs is reported. 100 held-out episodes per scenario.

## Pen-DHRL-TD (final checkpoint)

| Scenario | reward_avg | reward_avg_episodes | eplen_avg | captured_avg |
|---|---|---|---|---|
| corp_100hosts_dynamic (base) | 2.3216 | 259.12 | 111.61 | 29.63 |
| corp_100hosts_dynamic_varA | 0.2967 | 80.47 | 271.18 | 17.50 |
| corp_100hosts_dynamic_varB | 0.1688 | 46.04 | 272.69 | 13.25 |
| corp_100hosts_dynamic_bridge | 1.7425 | 205.48 | 117.93 | 24.67 |

## Pen-DHRL baseline (no distillation)

| Scenario | reward_avg | reward_avg_episodes | eplen_avg | captured_avg |
|---|---|---|---|---|
| corp_100hosts_dynamic (base) | 0.5428 | 217.11 | 400.0 | 29.59 |
| corp_100hosts_dynamic_varA | 0.1708 | 68.33 | 400.0 | 13.73 |
| corp_100hosts_dynamic_varB | 0.0778 | 31.12 | 400.0 | 9.39 |
| corp_100hosts_dynamic_bridge | 1.1470 | 161.96 | 141.20 | 23.36 |

Pen-DHRL-TD wins on every metric (`reward_avg`, `reward_avg_episodes`, `captured_avg`)
on every scenario, and terminates episodes well before the 400-step limit on all
four, while the baseline only does so on `bridge`. Since architecture and parameter
count are identical between the two, the gain is attributable to the LLM-teacher
distillation signal itself.

No baseline run was collected for the held-out `corp_100hosts_dynamic_test`
scenario, so it's omitted here (Pen-DHRL-TD alone: reward_avg=-0.0298,
reward_avg_episodes=-8.28, eplen_avg=277.84, captured_avg=9.20).

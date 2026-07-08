# Past-flow ControlNet final eval registration (2026-07-08)

This directory contains the design/review material for the past-flow ControlNet branch. This README records the final task eval results after the implementation and eval completed.

Protocol: `step=30000`, `video_keys=primary,left_view`, standard 4-suite LIBERO eval. Results were rechecked from 4090d eval logs on 2026-07-08 09:24 UTC. Deltas below roughly 6pp should be treated as eval noise unless repeated.

## Final results

### Past-flow ControlNet from-scratch

Run id: `qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary`

| suite | SR |
|---|---:|
| libero_spatial | 0.91 |
| libero_object | 0.94 |
| libero_goal | 0.91 |
| libero_10 | 0.67 |
| mean | 0.8575 |

Eval log root: `playground/Checkpoints/qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary/eval_step30000_primary_left_view/`.

### Coupled-Utonia scene-flow stage-2

Run id: `qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary`

| suite | SR |
|---|---:|
| libero_spatial | 0.89 |
| libero_object | 0.93 |
| libero_goal | 0.92 |
| libero_10 | 0.53 |
| mean | 0.8175 |

Eval log root: `playground/Checkpoints/qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary/eval_step30000_primary_left_view/`.

## Comparisons

| reference | mean SR | delta vs past-flow | note |
|---|---:|---:|---|
| plain scene-flow joint from-scratch | 0.8500 | +0.75pp | past-flow is not a clear improvement under the 6pp noise rule |
| plain scene-flow warmstart | 0.8275 | +3.00pp | still inside the noise band |
| previous Utonia top baseline | 0.8825 | -2.50pp | past-flow does not beat the best prior Utonia result |
| coupled-Utonia scene-flow stage-2 | 0.8175 | +4.00pp | coupled-Utonia did not win, mainly due to `libero_10` |

## Conclusion

Past-flow ControlNet finished at mean 0.8575. That is numerically the best scene-flow variant in this block, but only +0.75pp over the plain joint from-scratch 0.8500 baseline, so it should be recorded as neutral within noise rather than a positive result. Coupled-Utonia finished at 0.8175 and is clearly not the winning path in this run.

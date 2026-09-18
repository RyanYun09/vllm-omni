# Omni-DuplexEval CI Guard

Nightly end-to-end scoring regression for [Hothan/Omni-DuplexEval]
(RTD 6 subtasks + PR 3 subtasks), driven by `vllm bench omni-duplex-eval`
(generate → evaluate → summarize).

## Files

| File | Purpose |
| ------ | --------- |
| `test_omni_duplex_eval_ci.py` | Main pytest: hardcoded thresholds + 6 assertions |
| `omni_duplex_eval_ci_config.json` | Dataset / revision / sample IDs / judge |
| `conftest.py` | `judge_server` fixture (endpoint resolution only) |
| `README.md` | This document |

## Run locally

Judge server must be pre-started; this test never launches it:

```bash
# CUDA (H100 / B200), or NPU (A3)
VLLM_DUPLEX_EVAL_JUDGE_URL=http://127.0.0.1:8001 \
  pytest -s -v tests/e2e/accuracy/omni_duplex_eval/ \
    -m "full_model and cuda and (H100 or B200) and cards_1"
```

The `omni_server` fixture boots MiniCPM-o-4_5 with the community duplex
params (`tests/e2e/online_serving/helpers/minicpmo_4_5_duplex.py`).
`judge.base_url_env` (default `VLLM_DUPLEX_EVAL_JUDGE_URL`) overrides
`judge.base_url` from the JSON config.

## Baseline pre-run (Phase A)

Sample IDs in `sample_ids_per_split` are placeholders
(`RTD_world_knowledge_001`, ...). Replace them with real dataset IDs before
enabling the nightly gate:

1. Pin `dataset_revision` to a known-good commit and run
   `generate → evaluate → summarize` on the baseline commit
   `873e9bff7c545c5cda79fdb93a867f09e443b61d`.
2. Record `summarize_scores()` output (mean content / temporal / success).
3. Set `_MIN_*` thresholds = `mean - n * std` (start `n=2`) in
   `test_omni_duplex_eval_ci.py`.
4. Exclude known bad samples (generate hang / judge failure) via
   `exclude_ids`.

## Assertions (in order)

1. `protocol_pin == PROTOCOL_PIN` (track upstream protocol changes)
2. `summary["samples"]` == number of selected IDs
3. RTD `by_task` keys == 6 RTD subtasks
4. PR `by_task` keys == 3 PR subtasks
5. RTD mean content / temporal scores >= thresholds
6. PR `mean_all_success` >= threshold

# Four-model feature audit

## Findings in initial import

The core training/evaluation code and dependencies were present, and the GPU synthetic model check passed. This did not establish that historical launchers constituted a consistent workflow:

- The 1B `w3a16_g128_noalter.sh` actually names a w4 run and sets activation ratio 0.35 (about 4.1 bits for 2+2+2+2), not a 3-bit target.
- Other historical w3 scripts use ratio 0.15 (about 2.9 bits), rather than explicitly `--target_bit 3`.
- Training/sweep checkpoint paths differ, sometimes including manually moved checkpoint subdirectories. Historical 3B experiments mix base and Instruct models.
- Historical shell sweeps parse a shared tail of a tee log, do not check child exit codes, and may reuse earlier metrics or lose results due to log timing/length.
- The 2-bit MSB-only endpoint skips router statistics, producing N/A actual bits.
- Router stats claim multiple samples but originally report effective bits only from the last sample.
- Limited PPL evaluation uses an incorrect full-dataset denominator and evaluates limit+1 sequences.
- Packaging omitted termcolor and allowed Transformers versions incompatible with the active RoPE API.

## Destination-only fixes

The original buildq2 checkout remains unchanged. `MIGRATION_MANIFEST.json` is provenance for the initial byte-identical import, not a promise that later destination fixes retain those original hashes.

- `scripts/bitmoe/run.py` provides train/sweep presets for all four models. Training uses `--target_bit 3`, no ratio override and no `--onlymlp`. Both phases share the same base-model path and checkpoint directory. Paths can be overridden explicitly.
- `train_target3.sh` and `sweep_eval.sh` are thin shell entrypoints to this workflow.
- `main_bitmoe.py` writes atomic, finite JSON metrics on request. Sweep records per-point JSON/logs and incremental CSV, validates metrics and process exit status, and refuses to overwrite prior runs.
- Actual bits are averaged over all calibration samples and all MoE projections. They are calibration measurements, not PPL-test-token measurements or parameter-weighted storage rates. The exact MSB-only 2-bit endpoint is recorded without requiring router scores.
- `--seqlen` supports reduced validation. PPL sums token losses and divides by the actual number of evaluated next-token predictions. Positive `--limit` is the number of PPL sequences.
- Project metadata includes termcolor, Python >=3.10 and Transformers ==4.56.1. `dependencies.json` records installed versions satisfying all declared direct dependencies in omniquant. `requirements-validated.lock` records the tested direct versions. This is not a clean-room environment installation test.

## Validation scope

`gpu_smoke.py`: complete synthetic-model calibration and checkpoint restore, both stage backward passes and seven MoE projections. `metrics_checks.py`: uniform logits give exactly vocabulary-size PPL with limited and unlimited sequence counts. `workflow_checks.py`: all preset commands use the intended target/MoE mode, and failed or missing metrics never become successful CSV rows.

`real_model_checks.py`: real local pretrained model weights and cached WikiText, all layers, one epoch with two 32-token calibration samples, two 32-token PPL sequences, followed by fresh-process checkpoint restore and sweeps at 2, 3 and 8 bits. Run artifacts are copied to `validation/real-model-results/` after success.

These are functional tests. One tiny calibration epoch is not a converged 3-bit training run; the short PPL numbers must not be used as benchmark results. Production defaults remain 40 epochs, 128 samples, length 2048 and full WikiText PPL.

## Checkpoint architecture coverage

| Model preset | Decoder layers | Attention MoE projections | MLP MoE projections |
|---|---:|---:|---:|
| llama2-7b | 32 | 128 | 96 |
| llama3-1b | 16 | 64 | 48 |
| llama3-3b | 28 | 112 | 84 |
| llama3-8b | 32 | 128 | 96 |

Verified two router weight tensors for every q/k/v/o and gate/up/down projection in every layer of the real trained checkpoints. See `architecture-checks.json` and `collect_real_results.py`.

## Final result

All four real-model runs passed training, fresh-process restore, three sweep points and CSV generation. All 12 sweep rows have status `ok`; every layer contains all seven required router modules. Archived CSV paths are normalized relative to each archive directory; text logs have normalized line endings and trailing whitespace. See `real-model-results/summary.json`, each model's `bits-performance.csv`, JSON metrics, and train/eval logs.

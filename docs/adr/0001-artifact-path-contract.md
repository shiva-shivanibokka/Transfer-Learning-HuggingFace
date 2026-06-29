# ADR 0001 — Single source of truth for training-artifact paths

## Status
Accepted (2026-06-29).

## Context
The training scripts wrote model checkpoints to one directory layout while the
Gradio serving app read from a different hardcoded layout. They never matched,
so the app silently fell back to randomly-initialised weights and served
confident-but-wrong predictions with no error. The same drift affected the
calibration temperature file and the model-key registry (the app mapped display
names to keys like `vit` while the config registry used `vit_base`).

## Decision
All artifact locations are defined once in `src/utils/paths.py`. Both the
trainers (writers) and the app (reader) import these helpers. Model keys come
only from the `configs/*_config.py` registries; no module hardcodes a duplicate.
A unit test (`tests/test_paths.py`) asserts the app's display→key mapping
resolves against the registry, so the drift cannot silently return.

## Consequences
- Adding a model means updating the registry in one place.
- Changing the on-disk layout is a one-file change, covered by tests.
- Trade-off: a thin indirection layer instead of inline string paths — worth it
  given the failure was invisible in production (no error, just wrong output).

#!/bin/bash
# Compatibility wrapper for the unified Qwen3-SSD pipeline.
set -euo pipefail
exec ./run.sh --target s --test_corpora "emphassess" "$@"


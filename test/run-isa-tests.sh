#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
failures=0
passes=0

cargo build --release >/dev/null

for isatest in test/isa-tests/*; do
    test -x "$isatest" || continue
    name=$(basename "$isatest")
    log="logs/${name}.out"
    echo "Running $name ..."

    if ! RUST_LOG=debug ./target/release/cyclotron isa.toml --binary-path "$isatest" > "$log" 2>&1; then
        echo "::error [fail] $isatest"
        failures=$((failures+1))
    else
        echo "[pass] $isatest"
        passes=$((passes+1))
    fi
done

if [ $failures -ne 0 ]; then
    echo "isa summary: ${passes} passed, ${failures} failed"
    exit 1
else
    echo "isa summary: ${passes} passed, ${failures} failed"
fi

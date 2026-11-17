#!/usr/bin/env bash
set -euo pipefail
REPORT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPORT_DIR"
REPORT_BUILD="$(mktemp -d /tmp/me5411-report-build.XXXXXX)"
trap 'rm -rf "$REPORT_BUILD"' EXIT
if command -v tectonic >/dev/null 2>&1; then
    tectonic --outdir "$REPORT_BUILD" main.tex
elif command -v xelatex >/dev/null 2>&1; then
    xelatex -interaction=nonstopmode -halt-on-error -output-directory "$REPORT_BUILD" main.tex
    xelatex -interaction=nonstopmode -halt-on-error -output-directory "$REPORT_BUILD" main.tex
else
    echo 'Install Tectonic or XeLaTeX and ensure Arial is installed.' >&2
    exit 1
fi
cp "$REPORT_BUILD/main.pdf" "$REPORT_DIR/main.pdf"

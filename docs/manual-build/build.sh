#!/bin/sh
# Builds docs/Studio-Manual.pdf from docs/manual.md (needs pandoc and Node with Playwright).
set -e
cd "$(dirname "$0")/.."
pandoc manual.md -s --embed-resources --css manual-build/manual.css -o manual-build/manual.html
node manual-build/print.js manual-build/manual.html Studio-Manual.pdf

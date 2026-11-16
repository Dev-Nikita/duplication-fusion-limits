#!/usr/bin/env bash
# Builds the PUBLIC code-and-data archive (for Zenodo) ONLY from what Git tracks, so nothing private can slip in.
# Usage (after acceptance, or for a restricted-access Zenodo record before):  bash tools/make_public_release.sh v1.0
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tag="${1:-snapshot}"
bash tools/git_safety_check.sh >/dev/null || { echo "safety check failed"; exit 1; }
out="../paef-public-${tag}.zip"
git archive --format=zip --prefix="paef-${tag}/" -o "$out" HEAD
echo "written: $out ($(du -h "$out" | cut -f1)) — contents = tracked files of HEAD only"
unzip -l "$out" | tail -1

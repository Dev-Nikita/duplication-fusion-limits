#!/usr/bin/env bash
# Run BEFORE every commit/push:  bash tools/git_safety_check.sh
# Fails if anything private, large or binary-manuscript-like is staged.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
fail=0
staged=$(git diff --cached --name-only)
[ -z "$staged" ] && staged=$(git ls-files)
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    paper/*|*.tex|*.bib|*.pdf|*.docx|*.zip|*.tgz|*.tar.gz|data/*|data_external/*|testbed/out/*|tmp/*|docs/lit/*|docs/theory/*|docs/audit/*|submission/*)
      echo "BLOCKED (private/large path): $f"; fail=1;;
  esac
  case "$(basename "$f")" in
    *REVIEW_RESPONSE*|*EDITORIAL_*|*SUBMISSION_GATE*|*NOVELTY_AUDIT*|novelty-audit*|*PRIOR_ART*|FORMAL_*|THEOREM_VERIFICATION*|*MATHEMATICAL_VERIFICATION*|similarity_20*|*overlap-and-salami*|AI_USE*|cover_letter*|highlights*|RESEARCH_PLAN*|FEASIBILITY_EXTERNAL*)
      echo "BLOCKED (private document): $f"; fail=1;;
  esac
  [ "$f" = "tools/git_safety_check.sh" ] && continue
  if [ -f "$f" ]; then
    sz=$(wc -c < "$f")
    if [ "$sz" -gt 5000000 ]; then echo "BLOCKED (>5 MB): $f ($sz bytes)"; fail=1; fi
    if grep -Iq -E '\\documentclass|\\begin\{abstract\}|Dear Editor' "$f" 2>/dev/null; then echo "BLOCKED (manuscript text): $f"; fail=1; fi
    if grep -Iq -E '(ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)' "$f" 2>/dev/null; then echo "BLOCKED (secret-like string): $f"; fail=1; fi
  fi
done <<< "$staged"
if [ $fail -eq 0 ]; then echo "OK: $(echo "$staged" | wc -l | tr -d ' ') files checked, nothing private staged"; fi
exit $fail

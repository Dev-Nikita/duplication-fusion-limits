# What is deliberately NOT in Git, and where it lives

| Material | Location (local only) | Why not in Git |
|---|---|---|
| Manuscript source, PDF, figures for submission, cover letter, highlights | `paper/` | unpublished article |
| Unpublished theory notes, verification reports, proofs | `docs/theory/`, `docs/FORMAL_*`, `docs/THEOREM_VERIFICATION.md`, `docs/MATHEMATICAL_VERIFICATION_REPORT.md`, `docs/THEORY_CHANNEL_REGIMES.md` | priority before publication |
| Reviews, editorial assessments, submission strategy, AI-use draft | `docs/*REVIEW*`, `docs/EDITORIAL_*`, `docs/SUBMISSION_GATE*`, `docs/IF_READINESS_GATE*`, `docs/DECISION_*`, `docs/AI_USE_DISCLOSURE_DRAFT.md` | confidential |
| Comparisons with other own manuscripts | `docs/similarity_*.txt`, `docs/overlap-and-salami-guard.md`, `docs/PRIOR_ART_*`, `docs/NOVELTY_AUDIT*` | mention unpublished manuscripts |
| Literature PDFs | `docs/lit/` | copyright |
| Raw data (pcaps, raw logs), external datasets | `data/`, `data_external/`, `testbed/out/` | size (≈35 GB) and licences; derived results are in `results/` |
| Archives and sync bundles | `*.tgz`, `*.zip`, `FINAL_RESEARCH_RECORD_*` | binary, duplicates |

Backup of the private part (run monthly), to an external disk or encrypted cloud folder:
`tar czf ~/paef-private-$(date +%F).tgz paper docs && openssl enc -aes-256-cbc -pbkdf2 -in ~/paef-private-$(date +%F).tgz -out ~/paef-private-$(date +%F).tgz.enc`

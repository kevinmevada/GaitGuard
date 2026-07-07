# Deprecated utilities

One-off or manual tools kept for reference. **Not used by the pipeline, API, CI, or `run_local.py`.**

| Path | Purpose |
|------|---------|
| `scripts/audit_dataset_integrity.py` | Manual dataset integrity audit (not wired to CI) |
| `fall_risk_pipeline/scripts/export_subject_split.py` | CLI wrapper for `subject_split.ensure_subject_split_manifest` |
| `fall_risk_pipeline/scripts/validate_shell_scripts.sh` | Dev-only shellcheck helper (was used when condor/ existed) |

Safe to delete this folder if you do not need these tools.

Note: one-off git-history-rewrite scripts (`fix-github-attribution.ps1`,
`git-env-filter-fix-email.sh`, `git-msg-filter-strip-cursor.sh`) previously
listed here have been removed — they served a single historical cleanup
purpose and are not part of the reproducible research pipeline.

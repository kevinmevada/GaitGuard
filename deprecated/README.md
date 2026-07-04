# Deprecated utilities

One-off or manual tools kept for reference. **Not used by the pipeline, API, CI, or `run_local.py`.**

| Path | Purpose |
|------|---------|
| `scripts/git-env-filter-fix-email.sh` | Historical git filter to fix commit author emails |
| `scripts/git-msg-filter-strip-cursor.sh` | Historical git filter to strip Cursor attribution |
| `scripts/fix-github-attribution.ps1` | One-off PowerShell attribution cleanup |
| `scripts/audit_dataset_integrity.py` | Manual dataset integrity audit (not wired to CI) |
| `fall_risk_pipeline/scripts/export_subject_split.py` | CLI wrapper for `subject_split.ensure_subject_split_manifest` |
| `fall_risk_pipeline/scripts/validate_shell_scripts.sh` | Dev-only shellcheck helper (was used when condor/ existed) |

Safe to delete this folder if you do not need these tools.

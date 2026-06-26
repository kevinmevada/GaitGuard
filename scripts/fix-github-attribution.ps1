# Fix GitHub contributor attribution for Cursor-pushed commits.
#
# Problem: commits made with email "your-email@example.com" are attributed on GitHub
# to the Cursor cloud OAuth account (shown as renoschubert / second contributor).
#
# This script rewrites those commits to use mevadakevin@gmail.com (your GitHub account).
# After running, you must force-push main:
#   git push --force-with-lease origin main
#
# GitHub contributor graphs can take a few hours to refresh after the push.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$bad = git log main --format="%ae" | Select-String "your-email@example.com"
if (-not $bad) {
    Write-Host "No placeholder-email commits on main — nothing to rewrite."
    exit 0
}

Write-Host "Commits with placeholder email will be rewritten to Kevin Mevada <mevadakevin@gmail.com>"
Write-Host "Press Enter to continue or Ctrl+C to abort..."
Read-Host

$env:FILTER_BRANCH_SQUELCH_WARNING = "1"
$env:GIT_COMMITTER_NAME = "Kevin Mevada"
$env:GIT_COMMITTER_EMAIL = "mevadakevin@gmail.com"

git filter-branch -f --env-filter @'
if [ "$GIT_AUTHOR_EMAIL" = "your-email@example.com" ]; then
  export GIT_AUTHOR_EMAIL="mevadakevin@gmail.com"
  export GIT_AUTHOR_NAME="Kevin Mevada"
  export GIT_COMMITTER_EMAIL="mevadakevin@gmail.com"
  export GIT_COMMITTER_NAME="Kevin Mevada"
fi
'@ 463497c..HEAD

Write-Host ""
Write-Host "Done. Verify:"
git log main -2 --format="%an <%ae>"
Write-Host ""
Write-Host "Then update GitHub:"
Write-Host "  git push --force-with-lease origin main"

#!/bin/bash
# Fail if any file in the repository refers to a machine-specific location.
# Paths should go through bcrm/paths.py or R/paths.R instead.
cd "$(dirname "$0")/.."
pattern='(^|[^A-Za-z0-9._-])/(projects|scratch|pl/active|home)/|grahampeet|xsede'
found=0
git grep -n -I -E "$pattern" -- . ':!scripts/check_no_hardcoded_paths.sh' && found=1
# Untracked (not ignored) files too; skip grep entirely when there are none, or it reads stdin
untracked=$(git ls-files --others --exclude-standard | grep -vx 'scripts/check_no_hardcoded_paths.sh' || true)
if [ -n "$untracked" ]; then
    echo "$untracked" | tr '\n' '\0' | xargs -0 grep -H -n -I -E "$pattern" && found=1
fi
if [ "$found" = 1 ]; then
    echo "Machine-specific paths found (see above)." >&2
    exit 1
fi
echo "No machine-specific paths."

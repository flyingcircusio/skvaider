#!/bin/bash
set -euxo pipefail

cd "$(dirname "$0")"

current_branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$current_branch" != "main" ]; then
    set +x
    echo
    echo
    echo "ERROR: can only release from main. We are on $current_branch"
    echo
    exit 1
fi

changes=$(git status --porcelain)
if [ -n "$changes" ]; then
    set +x
    echo
    echo
    echo "ERROR: there are changes."
    echo
    git status
    exit 1
fi

uv run scriv collect

sed -i.bak '/- Nothing changed yet./ { N; d; }' CHANGES.md
rm -f CHANGES.md.bak

git add -A .
git status
PAGER= git diff --cached
echo "Press enter to commit, Ctrl-C to abort."
read
git commit -m "Prepare changelog for release"

uv run --group release fullrelease

#!/usr/bin/env bash
# Build/refresh the `public` branch from the committed tree of `main`, WITHOUT main's history:
#   - drops private/ and the `@private/...` import line in CLAUDE.md
#   - refuses to build if it finds IPs, cloud IDs, tokens or keys
#   - each run adds one commit on top of the previous `public` commit (first run: a root commit)
# Usage: scripts/make_public.sh "commit message"      then push ONLY this branch to the public remote:
#        git push <public-remote> public:main
set -euo pipefail
cd "$(dirname "$0")/.."
MSG=${1:?usage: scripts/make_public.sh "commit message"}
git diff --quiet && git diff --cached --quiet || { echo "commit or stash your changes first"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "run from main"; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir "$TMP/tree"
git archive HEAD | tar -x -C "$TMP/tree"
rm -rf "$TMP/tree/private"
grep -v '^@private/' "$TMP/tree/CLAUDE.md" > "$TMP/claude" && mv "$TMP/claude" "$TMP/tree/CLAUDE.md"

LEAKS='([0-9]{1,3}\.){3}[0-9]{1,3}/32|89\.169\.|computeinstance-|project-e00|vpc(subnet|allocation|network|securitygroup)-|computedisk-|hf_[A-Za-z0-9]{30,}|ssh-ed25519 |BEGIN [A-Z ]*PRIVATE KEY|github-personal|private/'
if grep -rInE "$LEAKS" "$TMP/tree" --exclude=uv.lock --exclude=make_public.sh; then
  echo "^^^ possible private data in the public tree: fix on main first"; exit 1
fi

export GIT_INDEX_FILE="$TMP/index"
git --work-tree="$TMP/tree" add -A
TREE=$(git write-tree)
PARENT=$(git rev-parse -q --verify refs/heads/public || true)
if [ -n "$PARENT" ] && [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then echo "public is already up to date"; exit 0; fi
COMMIT=$(git commit-tree "$TREE" ${PARENT:+-p "$PARENT"} -m "$MSG")
git update-ref refs/heads/public "$COMMIT"
echo "public -> $COMMIT ($(git ls-tree -r --name-only public | wc -l | tr -d ' ') files)"

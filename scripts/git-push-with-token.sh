#!/usr/bin/env bash
# Push to GitHub using a Personal Access Token (no username/password prompts).
# The token is stored in the remote URL so future "git push" never asks for a password.
#
# Usage:
#   1. Put your token in a file (optional, so you don't type it each time):
#        echo "ghp_yourToken" > .github-token
#        chmod 600 .github-token
#   2. Run (replace yourusername with your GitHub username):
#        GITHUB_USER=yourusername GITHUB_PAT=ghp_xxxx ./scripts/git-push-with-token.sh
#      Or if you created .github-token:
#        GITHUB_USER=yourusername GITHUB_PAT=$(cat .github-token) ./scripts/git-push-with-token.sh
#   3. To overwrite the remote if it has different history (use with care):
#        GITHUB_USER=yourusername GITHUB_PAT=xxx ./scripts/git-push-with-token.sh --force

set -e
cd "$(dirname "$0")/.."

FORCE_PUSH=false
for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE_PUSH=true
done

GITHUB_USER="${GITHUB_USER:-}"
GITHUB_PAT="${GITHUB_PAT:-}"

# Optional: read PAT from file if not set
if [ -z "$GITHUB_PAT" ] && [ -f ".github-token" ]; then
  GITHUB_PAT=$(cat .github-token)
fi

if [ -z "$GITHUB_USER" ] || [ -z "$GITHUB_PAT" ]; then
  echo "Set your GitHub username and PAT (Personal Access Token)."
  echo ""
  echo "Option 1 - run with env vars:"
  echo "  GITHUB_USER=yourusername GITHUB_PAT=ghp_xxxx ./scripts/git-push-with-token.sh"
  echo ""
  echo "Option 2 - put the key in a file once (recommended):"
  echo "  echo \"ghp_yourToken\" > .github-token"
  echo "  chmod 600 .github-token"
  echo "  GITHUB_USER=yourusername GITHUB_PAT=\$(cat .github-token) ./scripts/git-push-with-token.sh"
  echo ""
  echo "Add --force to overwrite remote if it has different history."
  exit 1
fi

# Remove existing origin so we can set it with the token in the URL
git remote remove origin 2>/dev/null || true
# This URL embeds the token so git will never ask for username/password
git remote add origin "https://${GITHUB_PAT}@github.com/${GITHUB_USER}/trading-bot.git"

if [ "$FORCE_PUSH" = true ]; then
  git push -u origin main --force
else
  if ! git pull origin main --rebase 2>/dev/null; then
    echo "Remote has different history. To overwrite it with your local branch, run:"
    echo "  GITHUB_USER=$GITHUB_USER GITHUB_PAT=\$(cat .github-token) ./scripts/git-push-with-token.sh --force"
    exit 1
  fi
  git push -u origin main
fi

echo "Push done. Future 'git push' from this repo will use the same remote (no password prompt)."

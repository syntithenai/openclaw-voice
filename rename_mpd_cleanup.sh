#!/usr/bin/env bash
# Completes the mpd → native music rename/cleanup tasks that were not
# finished in the previous session.
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
SKILLS="/home/stever/projects/openclawstuff/.openclaw/workspace-voice/skills"

ok()  { echo "  ✓ $*"; }
skip(){ echo "  – skip: $*"; }
fail(){ echo "  ✗ $*" >&2; }

echo ""
echo "=== openclaw-voice mpd cleanup ==="
echo ""

# ── 1. Root-level script / test file renames ──────────────────────────────
echo "── Root-level file renames ──"

for pair in \
  "setup_mpd.sh:setup_native_music.sh" \
  "fix_mpd.sh:fix_native_music.sh" \
  "validate_mpd_integration.py:validate_native_music_integration.py" \
  "test_mpd_orchestrator.py:test_native_music_orchestrator.py"
do
  src="${pair%%:*}"
  dst="${pair##*:}"
  if [[ -f "$BASE/$src" ]]; then
    mv "$BASE/$src" "$BASE/$dst"
    ok "$src  →  $dst"
  elif [[ -f "$BASE/$dst" ]]; then
    skip "$dst already exists"
  else
    skip "$src not found"
  fi
done

echo ""

# ── 2. Rename mpd_client.py → native_backend.py and fix imports ───────────
echo "── orchestrator/music/mpd_client.py → native_backend.py ──"

MUSIC_DIR="$BASE/orchestrator/music"
OLD_PY="$MUSIC_DIR/mpd_client.py"
NEW_PY="$MUSIC_DIR/native_backend.py"

if [[ -f "$OLD_PY" ]]; then
  mv "$OLD_PY" "$NEW_PY"
  ok "mpd_client.py  →  native_backend.py"
elif [[ -f "$NEW_PY" ]]; then
  skip "native_backend.py already exists"
fi

# Fix import in orchestrator/music/__init__.py
INIT_PY="$MUSIC_DIR/__init__.py"
if [[ -f "$INIT_PY" ]] && grep -q 'mpd_client' "$INIT_PY"; then
  sed -i 's/from \.mpd_client /from .native_backend /g' "$INIT_PY"
  ok "__init__.py imports updated"
else
  skip "__init__.py already clean or not found"
fi

# Fix import in orchestrator/music/native_client.py
NATIVE_PY="$MUSIC_DIR/native_client.py"
if [[ -f "$NATIVE_PY" ]] && grep -q 'mpd_client' "$NATIVE_PY"; then
  sed -i 's/from \.mpd_client /from .native_backend /g' "$NATIVE_PY"
  # Also update the comment on the same line if present
  sed -i 's/lives in mpd_client\.py/lives in native_backend.py/g' "$NATIVE_PY"
  ok "native_client.py imports updated"
else
  skip "native_client.py already clean or not found"
fi

echo ""

# ── 3. Rename skills/mpd → skills/music_player ────────────────────────────
echo "── skills/mpd  →  skills/music_player ──"

MPD_SKILL="$SKILLS/mpd"
MP_SKILL="$SKILLS/music_player"

if [[ -d "$MPD_SKILL" ]]; then
  mv "$MPD_SKILL" "$MP_SKILL"
  ok "skills/mpd  →  skills/music_player"
elif [[ -d "$MP_SKILL" ]]; then
  skip "skills/music_player already exists"
else
  skip "skills/mpd not found"
fi

# Remove test_mpd.sh from skills (no longer needed)
TEST_SH="$MP_SKILL/scripts/test_mpd.sh"
if [[ -f "$TEST_SH" ]]; then
  rm "$TEST_SH"
  ok "removed skills/music_player/scripts/test_mpd.sh"
else
  skip "test_mpd.sh already removed"
fi

echo ""

# ── 4. Remove empty docker/mpd directory ──────────────────────────────────
echo "── docker/mpd directory ──"

DOCKER_MPD="$BASE/docker/mpd"
if [[ -d "$DOCKER_MPD" ]]; then
  if [[ -z "$(ls -A "$DOCKER_MPD")" ]]; then
    rmdir "$DOCKER_MPD"
    ok "removed empty docker/mpd/"
  else
    fail "docker/mpd/ is not empty — skipping (manual review needed)"
    ls "$DOCKER_MPD"
  fi
else
  skip "docker/mpd/ already removed"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  • Review skills/music_player/mpd.md and rename/update as needed"
echo "  • Review skills/music_player/scripts/mpd_remote.py for backend references"
echo "  • Restart the orchestrator to pick up the native_backend.py rename"
echo ""

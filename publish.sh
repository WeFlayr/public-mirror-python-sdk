#!/usr/bin/env bash
# publish.sh — Build and publish the Weflayr SDK to PyPI or TestPyPI
#
# Usage:
#   ./publish.sh            → publish to PyPI (production)
#   ./publish.sh --test     → publish to TestPyPI
#   ./publish.sh --build    → build only, skip upload

set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()    { echo -e "${BLUE}[weflayr]${NC} $*"; }
success(){ echo -e "${GREEN}[weflayr]${NC} $*"; }
warn()   { echo -e "${YELLOW}[weflayr]${NC} $*"; }
error()  { echo -e "${RED}[weflayr]${NC} $*" >&2; exit 1; }

# ─── Parse arguments ──────────────────────────────────────────────────────────
TARGET="pypi"
BUILD_ONLY=false

for arg in "$@"; do
  case $arg in
    --test)   TARGET="testpypi" ;;
    --build)  BUILD_ONLY=true ;;
    --help|-h)
      echo "Usage: ./publish.sh [--test] [--build]"
      echo ""
      echo "  (no flag)   Build and publish to PyPI (production)"
      echo "  --test      Build and publish to TestPyPI"
      echo "  --build     Build only, skip upload"
      exit 0
      ;;
    *) error "Unknown argument: $arg. Use --help for usage." ;;
  esac
done

# ─── Sanity checks ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command -v uv   >/dev/null 2>&1 || error "uv is not installed. See https://docs.astral.sh/uv/"
command -v twine >/dev/null 2>&1 || { warn "twine not found, installing via uv..."; uv add --dev twine; }

# ─── Step 1: Clean previous builds ───────────────────────────────────────────
log "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
success "dist/ cleaned."

# ─── Step 2: Run tests ────────────────────────────────────────────────────────
log "Running tests..."
if uv run pytest --tb=short -q; then
  success "All tests passed."
else
  error "Tests failed. Fix them before publishing."
fi

# ─── Step 3: Build ────────────────────────────────────────────────────────────
log "Building package..."
uv run python -m build
success "Build complete. Artifacts:"
ls -lh dist/

# ─── Step 4: Check with twine ─────────────────────────────────────────────────
log "Checking package integrity with twine..."
uv run twine check dist/*
success "Package check passed."

# ─── Step 5: Upload ───────────────────────────────────────────────────────────
if $BUILD_ONLY; then
  warn "Build-only mode — skipping upload."
  exit 0
fi

if [ "$TARGET" = "testpypi" ]; then
  warn "Publishing to TestPyPI..."
  uv run twine upload --repository testpypi dist/*
  echo ""
  success "Published to TestPyPI!"
  echo -e "  ${BLUE}→ View:${NC}    https://test.pypi.org/project/weflayr/"
  echo -e "  ${BLUE}→ Install:${NC} pip install --index-url https://test.pypi.org/simple/ weflayr"
else
  warn "Publishing to PyPI (production)..."
  echo -e "${YELLOW}Are you sure you want to publish to PyPI? [y/N]${NC} \c"
  read -r confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    warn "Aborted."
    exit 0
  fi
  uv run twine upload dist/*
  echo ""
  success "Published to PyPI!"
  echo -e "  ${BLUE}→ View:${NC}    https://pypi.org/project/weflayr/"
  echo -e "  ${BLUE}→ Install:${NC} pip install weflayr"
fi

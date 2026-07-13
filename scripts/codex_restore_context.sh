#!/usr/bin/env bash
set -euo pipefail

readonly ARCHIVE_REPO="eacsai/starvla-project-archive"
readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly DEFAULT_ARCHIVE_DIR="$(dirname -- "${REPO_ROOT}")/starvla-project-archive"
readonly ARCHIVE_DIR="${STARVLA_ARCHIVE_DIR:-${DEFAULT_ARCHIVE_DIR}}"

READ_ONLY=0
if [[ "${1:-}" == "--read-only" ]]; then
  READ_ONLY=1
  shift
fi

if [[ "$#" -ne 0 ]]; then
  printf 'Usage: %s [--read-only]\n' "$0" >&2
  exit 2
fi

readonly READ_ONLY

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

clone_archive() {
  if [[ "${READ_ONLY}" -eq 1 ]]; then
    fail "archive is missing in read-only mode: ${ARCHIVE_DIR}"
  fi

  printf 'Private project archive is missing at %s\n' "${ARCHIVE_DIR}"
  mkdir -p "$(dirname -- "${ARCHIVE_DIR}")"

  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    if gh repo clone "${ARCHIVE_REPO}" "${ARCHIVE_DIR}"; then
      return
    fi
    printf 'WARNING: GitHub CLI clone failed; trying authenticated HTTPS.\n' >&2
    if [[ -e "${ARCHIVE_DIR}" ]]; then
      fail "failed clone left the archive path in place: ${ARCHIVE_DIR}"
    fi
  fi

  if GIT_TERMINAL_PROMPT=0 git clone \
    "https://github.com/${ARCHIVE_REPO}.git" "${ARCHIVE_DIR}"; then
    return
  fi

  fail "cannot clone the private archive. Authenticate GitHub CLI as eacsai, then rerun: gh auth login"
}

validate_archive_remote() {
  local origin_url
  origin_url="$(git -C "${ARCHIVE_DIR}" remote get-url origin 2>/dev/null)" || \
    fail "archive has no origin remote: ${ARCHIVE_DIR}"

  case "${origin_url}" in
    "https://github.com/${ARCHIVE_REPO}"|\
    "https://github.com/${ARCHIVE_REPO}.git"|\
    "git@github.com:${ARCHIVE_REPO}.git"|\
    "ssh://git@github.com/${ARCHIVE_REPO}.git")
      ;;
    *)
      fail "archive origin does not match ${ARCHIVE_REPO}: ${origin_url}"
      ;;
  esac
}

update_archive() {
  local branch

  if [[ "${READ_ONLY}" -eq 1 ]]; then
    return
  fi

  if [[ -n "$(git -C "${ARCHIVE_DIR}" status --porcelain)" ]]; then
    printf 'WARNING: archive has local changes; leaving it untouched.\n' >&2
    return
  fi

  branch="$(git -C "${ARCHIVE_DIR}" branch --show-current)"
  if [[ "${branch}" != "main" ]]; then
    printf 'WARNING: archive checkout is not main; leaving it untouched.\n' >&2
    return
  fi

  if ! git -C "${ARCHIVE_DIR}" fetch origin main; then
    printf 'WARNING: archive update unavailable; using the existing local copy.\n' >&2
    return
  fi

  if ! git -C "${ARCHIVE_DIR}" merge --ff-only origin/main; then
    printf 'WARNING: archive main diverged; using the existing local copy.\n' >&2
  fi
}

if [[ ! -d "${ARCHIVE_DIR}" ]]; then
  clone_archive
fi

git -C "${ARCHIVE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "archive path is not a Git repository: ${ARCHIVE_DIR}"

validate_archive_remote
update_archive

required_files=(
  "README.md"
  "CONTINUATION.md"
  "project_memory/codex_handoffs/handoff-2026-07-13T03-17-48Z.md"
  "reproduction/DATASETS_AND_WEIGHTS.md"
  "reproduction/EXCLUDED_ASSETS.md"
  "manifests/code_repositories.tsv"
  "results/starvla_experiments/README.md"
)

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "${ARCHIVE_DIR}/${path}" ]]; then
    printf 'MISSING: %s\n' "${ARCHIVE_DIR}/${path}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  fail "private archive is incomplete; inspect its branch and checkout"
fi

printf '\nStarVLA context archive ready\n'
printf '  code root:    %s\n' "${REPO_ROOT}"
printf '  code branch:  %s\n' "$(git -C "${REPO_ROOT}" branch --show-current)"
printf '  code commit:  %s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
printf '  archive root: %s\n' "${ARCHIVE_DIR}"
printf '  archive head: %s\n' "$(git -C "${ARCHIVE_DIR}" rev-parse HEAD)"

printf '\nRead these files before starting project work:\n'
for path in "${required_files[@]}"; do
  printf '  %s\n' "${ARCHIVE_DIR}/${path}"
done

printf '\nTask-specific memory roots:\n'
printf '  papers and notes: %s\n' "${ARCHIVE_DIR}/research"
printf '  experiment data:  %s\n' "${ARCHIVE_DIR}/results"
printf '  prior decisions:  %s\n' "${ARCHIVE_DIR}/project_memory"
printf '  reconstruction:   %s\n' "${ARCHIVE_DIR}/reproduction"

printf '\nRecovery completed. Read AGENTS.md and docs/CODEX_PROJECT_RECOVERY.md in the code repository too.\n'

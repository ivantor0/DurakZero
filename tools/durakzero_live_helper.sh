#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-${PROJECT_ROOT}/model.tar}
DEVICE=${DEVICE:-cpu}
PLAYER_ID=${PLAYER_ID:-}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/durak_live_logs}
SOURCE=${SOURCE:-pyshark}
INTERFACE=${INTERFACE:-waydroid0}
IP_FILTER=${IP_FILTER:-}
LOG_FILE=${LOG_FILE:-}

ARGS=("--checkpoint" "${CHECKPOINT_PATH}" "--device" "${DEVICE}" "--source" "${SOURCE}")

if [[ -n "${PLAYER_ID}" ]]; then
  ARGS+=("--player-id" "${PLAYER_ID}")
fi

if [[ -n "${LOG_DIR}" ]]; then
  ARGS+=("--log-dir" "${LOG_DIR}")
fi

case "${SOURCE}" in
  file)
    if [[ -z "${LOG_FILE}" ]]; then
      echo "LOG_FILE must be set when SOURCE=file" >&2
      exit 1
    fi
    ARGS+=("--log-file" "${LOG_FILE}")
    ;;
  pyshark)
    ARGS+=("--interface" "${INTERFACE}")
    if [[ -n "${IP_FILTER}" ]]; then
      ARGS+=("--ip-filter" "${IP_FILTER}")
    fi
    ;;
  *)
    echo "Unsupported SOURCE '${SOURCE}'. Use 'pyshark' or 'file'." >&2
    exit 1
    ;;
 esac

exec python "${SCRIPT_DIR}/durak_live_helper.py" "${ARGS[@]}"

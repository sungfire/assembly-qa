#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
MODEL_FILE_ID="1dVJjm3ZjUOXYpeQxXXowu7gnhWDfb6gV"
MODEL_ARCHIVE="${REPO_ROOT}/model.zip"
EXTRACT_DIR="${REPO_ROOT}/.model_extract"
TARGET_BASE="/workspace"
TARGET_DIR="${TARGET_BASE}/2.AI학습모델파일/1. 질의응답/nia15-polyglot-5.8b-koalpaca-v1.1b-qna-best"
REQUIREMENTS_FILE="${REPO_ROOT}/backend/requirements.txt"
VENV_DIR="${REPO_ROOT}/.venv"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' command is required but not found." >&2
        exit 1
    fi
}

download_model() {
    echo "Downloading model archive from Google Drive..."
    local cookie_file="${REPO_ROOT}/.gdrive_cookie"
    rm -f "${MODEL_ARCHIVE}" "${cookie_file}"
    curl -sc "${cookie_file}" "https://drive.google.com/uc?export=download&id=${MODEL_FILE_ID}" >/dev/null
    local confirm_code
    confirm_code="$(awk '/_warning_/ {print $NF}' "${cookie_file}")"
    curl -Lb "${cookie_file}" "https://drive.google.com/uc?export=download&confirm=${confirm_code}&id=${MODEL_FILE_ID}" -o "${MODEL_ARCHIVE}"
    rm -f "${cookie_file}"
}

unpack_model() {
    echo "Extracting model archive..."
    rm -rf "${EXTRACT_DIR}"
    mkdir -p "${EXTRACT_DIR}"
    unzip -q "${MODEL_ARCHIVE}" -d "${EXTRACT_DIR}"
}

deploy_model() {
    echo "Deploying model files to ${TARGET_DIR} ..."
    mkdir -p "${TARGET_BASE}"
    if [ -d "${TARGET_DIR}" ]; then
        echo "Removing existing model directory at ${TARGET_DIR}"
        rm -rf "${TARGET_DIR}"
    fi
    rsync -a "${EXTRACT_DIR}/2.AI학습모델파일/" "${TARGET_BASE}/2.AI학습모델파일/"
}

setup_python_env() {
    echo "Setting up Python virtual environment..."
    if [ ! -d "${VENV_DIR}" ]; then
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r "${REQUIREMENTS_FILE}"
    deactivate
}

clean_up() {
    echo "Cleaning up temporary files..."
    rm -f "${MODEL_ARCHIVE}"
    rm -rf "${EXTRACT_DIR}"
}

main() {
    require_command curl
    require_command unzip
    require_command python3
    require_command rsync

    download_model
    unpack_model
    deploy_model
    setup_python_env
    clean_up

    echo "Setup completed successfully."
}

main "$@"

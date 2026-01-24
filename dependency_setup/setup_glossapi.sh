#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="vanilla"
PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${GLOSSAPI_VENV:-}"
DOWNLOAD_DEEPSEEK=0
DEEPSEEK_ROOT="${DEEPSEEK_ROOT:-${REPO_ROOT}/deepseek-ocr}"
MINERU_COMMAND="${GLOSSAPI_MINERU_COMMAND:-}"
DOWNLOAD_MINERU_MODELS=0
MINERU_MODELS_REPO="${MINERU_MODELS_REPO:-opendatalab/PDF-Extract-Kit-1.0}"
RUN_TESTS=0
RUN_SMOKE=0

usage() {
  cat <<'EOF'
Usage: setup_glossapi.sh [options]

Options:
  --mode MODE            Environment profile: vanilla, rapidocr, deepseek, mineru (default: vanilla)
  --venv PATH            Target virtual environment path
  --python PATH          Python executable to use when creating the venv
  --download-deepseek    Fetch DeepSeek-OCR weights (only meaningful for --mode deepseek)
  --weights-dir PATH     Destination directory for DeepSeek weights (default: $REPO_ROOT/deepseek-ocr)
  --download-mineru-models
                         Download MinerU model bundle into dependency_setup/mineru_models
  --mineru-command PATH  Path to magic-pdf binary (optional; stored in GLOSSAPI_MINERU_COMMAND)
  --run-tests            Run pytest -q after installation
  --smoke-test           Run dependency_setup/deepseek_gpu_smoke.py (deepseek mode only)
  --help                 Show this help message
EOF
}

while (( "$#" )); do
  case "$1" in
    --mode)
      shift || { echo "--mode requires a value" >&2; exit 1; }
      MODE="${1:-}"
      ;;
    --venv)
      shift || { echo "--venv requires a path" >&2; exit 1; }
      VENV_PATH="${1:-}"
      ;;
    --python)
      shift || { echo "--python requires a path" >&2; exit 1; }
      PYTHON_BIN="${1:-}"
      ;;
    --download-deepseek)
      DOWNLOAD_DEEPSEEK=1
      ;;
    --weights-dir)
      shift || { echo "--weights-dir requires a path" >&2; exit 1; }
      DEEPSEEK_ROOT="${1:-}"
      ;;
    --download-mineru-models)
      DOWNLOAD_MINERU_MODELS=1
      ;;
    --mineru-command)
      shift || { echo "--mineru-command requires a path" >&2; exit 1; }
      MINERU_COMMAND="${1:-}"
      ;;
    --run-tests)
      RUN_TESTS=1
      ;;
    --smoke-test)
      RUN_SMOKE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift || true
done

case "${MODE}" in
  vanilla|rapidocr|deepseek|mineru) ;;
  *)
    echo "Invalid mode '${MODE}'. Expected vanilla, rapidocr, deepseek, or mineru." >&2
    exit 1
    ;;
esac

if [[ -z "${VENV_PATH}" ]]; then
  VENV_PATH="${REPO_ROOT}/.venv_glossapi_${MODE}"
fi

REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-glossapi-${MODE}.txt"
if [[ "${MODE}" == "rapidocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-glossapi-rapidocr-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  fi
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Requirements file not found for mode ${MODE}: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

info()  { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; exit 1; }

ensure_venv() {
  if [[ ! -d "${VENV_PATH}" ]]; then
    info "Creating virtual environment at ${VENV_PATH}"
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  else
    info "Reusing existing virtual environment at ${VENV_PATH}"
  fi
}

pip_run() {
  "${VENV_PATH}/bin/pip" "$@"
}

python_run() {
  "${VENV_PATH}/bin/python" "$@"
}

patch_mineru_unimernet_cache_position() {
  python_run - <<'PY'
import re
from pathlib import Path

try:
  import magic_pdf.model.sub_modules.mfr.unimernet.unimernet_hf.unimer_mbart.modeling_unimer_mbart as mod
except Exception as exc:
  print(f"[warn] Unable to import Unimernet MBart module: {exc}")
  raise SystemExit(0)

path = Path(mod.__file__)
text = path.read_text(encoding="utf-8")

cache_sig_done = "cache_position" in text and "**kwargs" in text

pattern = "count_gt: Optional[torch.LongTensor] = None,\n"
replacement = (
  "count_gt: Optional[torch.LongTensor] = None,\n"
  "        cache_position: Optional[torch.LongTensor] = None,\n"
  "        **kwargs,\n"
)

if not cache_sig_done:
  if pattern not in text:
    print("[warn] Could not find Unimernet forward signature to patch.")
  else:
    text = text.replace(pattern, replacement, 1)
    print("[info] Patched Unimernet forward signature.")

cache_line = "past_key_values_length = past_key_values[0][0].shape[2]"
if cache_line in text:
  cache_block = (
    "past_key_values_length = 0\n"
    "        if past_key_values is not None:\n"
    "            if hasattr(past_key_values, \"get_seq_length\"):\n"
    "                past_key_values_length = past_key_values.get_seq_length() or 0\n"
    "            elif hasattr(past_key_values, \"to_legacy_cache\"):\n"
    "                legacy = past_key_values.to_legacy_cache()\n"
    "                past_key_values_length = legacy[0][0].shape[2] if legacy else 0\n"
    "            elif hasattr(past_key_values, \"past_key_values\"):\n"
    "                legacy = past_key_values.past_key_values\n"
    "                past_key_values_length = legacy[0][0].shape[2] if legacy else 0\n"
    "            else:\n"
    "                past_key_values_length = past_key_values[0][0].shape[2]"
  )
  text = text.replace(cache_line, cache_block, 1)
  print("[info] Patched Unimernet cache handling for EncoderDecoderCache.")

path.write_text(text, encoding="utf-8")
print(f"[info] Unimernet patch complete: {path}")
PY
}

patch_mineru_unimernet_sdpa_none_cache() {
  python_run - <<'PY'
from pathlib import Path

try:
  import magic_pdf.model.sub_modules.mfr.unimernet.unimernet_hf.unimer_mbart.modeling_unimer_mbart as mod
except Exception as exc:
  print(f"[warn] Unable to import Unimernet MBart module: {exc}")
  raise SystemExit(0)

path = Path(mod.__file__)
text = path.read_text(encoding="utf-8")

marker = "is_cross_attention = key_value_states is not None\n"
guard = (
  "is_cross_attention = key_value_states is not None\n\n"
  "        if past_key_value is not None and (past_key_value[0] is None or past_key_value[1] is None):\n"
  "            past_key_value = None\n"
)

if guard in text:
  print("[info] Unimernet SDPA None-cache guard already applied.")
  raise SystemExit(0)

if marker not in text:
  print("[warn] Could not find SDPA attention marker to patch.")
  raise SystemExit(0)

text = text.replace(marker, guard, 1)
path.write_text(text, encoding="utf-8")
print(f"[info] Patched Unimernet SDPA None-cache guard in {path}")
PY
}

patch_mineru_paddleocr_config() {
  python_run - <<'PY'
from pathlib import Path

try:
  import magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorchocr.utils.resources as res
except Exception as exc:
  print(f"[warn] Unable to import PaddleOCR resources: {exc}")
  raise SystemExit(0)

path = Path(res.__file__).parent / "models_config.yml"
if not path.exists():
  print("[warn] PaddleOCR models_config.yml not found.")
  raise SystemExit(0)

text = path.read_text(encoding="utf-8")
text = text.replace("det: ch_PP-OCRv3_det_infer.pth", "det: ch_PP-OCRv5_det_infer.pth")
text = text.replace("rec: ch_PP-OCRv4_rec_infer.pth", "rec: ch_PP-OCRv5_rec_infer.pth")
path.write_text(text, encoding="utf-8")
print(f"[info] Patched PaddleOCR model config in {path}")
PY
}

resolve_mineru_cmd() {
  local cmd_path="${MINERU_COMMAND:-}"
  if [[ -n "${cmd_path}" ]]; then
    echo "${cmd_path}"
    return 0
  fi
  if [[ -n "${VENV_PATH}" && -x "${VENV_PATH}/bin/magic-pdf" ]]; then
    echo "${VENV_PATH}/bin/magic-pdf"
    return 0
  fi
  if command -v magic-pdf >/dev/null 2>&1; then
    command -v magic-pdf
    return 0
  fi
  echo ""
  return 0
}

download_deepseek_weights() {
  local root="$1"
  local target="${root}/DeepSeek-OCR"

  if [[ -d "${target}" ]]; then
    info "DeepSeek-OCR weights already present at ${target}"
    return 0
  fi

  mkdir -p "${root}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    info "Downloading DeepSeek weights with huggingface-cli (this may take a while)"
    huggingface-cli download deepseek-ai/DeepSeek-OCR \
      --repo-type model \
      --include "DeepSeek-OCR/*" \
      --local-dir "${target}" \
      --local-dir-use-symlinks False || warn "huggingface-cli download failed; falling back to git-lfs"
  fi

  if [[ ! -d "${target}" ]]; then
    if command -v git >/dev/null 2>&1; then
      if ! command -v git-lfs >/dev/null 2>&1; then
        warn "git-lfs not available; install git-lfs to clone DeepSeek weights via git."
      else
        info "Cloning DeepSeek weights via git-lfs"
        git lfs install --skip-repo >/dev/null 2>&1 || true
        git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR "${target}"
      fi
    else
      warn "Neither huggingface-cli nor git found; skipping DeepSeek weight download."
    fi
  fi

  if [[ ! -d "${target}" ]]; then
    warn "DeepSeek weights were not downloaded. Set DEEPSEEK_ROOT manually once acquired."
  fi
}

download_mineru_models() {
  local target_dir="$1"
  info "Downloading MinerU models from ${MINERU_MODELS_REPO} to ${target_dir} (this may take a while)"
  pip_run install "huggingface_hub" || warn "huggingface_hub install failed; cannot download MinerU models."
  python_run - <<PY
import os
from huggingface_hub import snapshot_download

target_dir = r"${target_dir}"
os.makedirs(target_dir, exist_ok=True)
snapshot_download(
    repo_id="${MINERU_MODELS_REPO}",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    max_workers=20,
)
print("MinerU model download complete.")
PY
}

ensure_venv
info "Upgrading pip tooling"
pip_run install --upgrade pip wheel setuptools

info "Installing ${MODE} requirements from $(basename "${REQUIREMENTS_FILE}")"
pip_run install -r "${REQUIREMENTS_FILE}"

info "Installing glossapi in editable mode"
pip_run install -e "${REPO_ROOT}" --no-deps

info "Building Rust extensions via editable installs"
pip_run install -e "${REPO_ROOT}/rust/glossapi_rs_cleaner"
pip_run install -e "${REPO_ROOT}/rust/glossapi_rs_noise"

if [[ "${MODE}" == "deepseek" ]]; then
  export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT="${DEEPSEEK_ROOT}/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH="${DEEPSEEK_ROOT}/libjpeg-turbo/lib"
  export GLOSSAPI_DEEPSEEK_ALLOW_STUB=0
  export LD_LIBRARY_PATH="${GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

  if [[ "${DOWNLOAD_DEEPSEEK}" -eq 1 ]]; then
    download_deepseek_weights "${DEEPSEEK_ROOT}"
  else
    warn "DeepSeek weights not downloaded (use --download-deepseek to fetch automatically)."
  fi
fi

if [[ "${MODE}" == "mineru" ]]; then
  info "Patching Unimernet for transformers cache_position compatibility"
  patch_mineru_unimernet_cache_position || warn "Unimernet patch failed; math formula recognition may error on newer transformers."
  info "Patching Unimernet SDPA cache handling"
  patch_mineru_unimernet_sdpa_none_cache || warn "Unimernet SDPA cache patch failed."
  info "Patching PaddleOCR model mapping"
  patch_mineru_paddleocr_config || warn "PaddleOCR model config patch failed."

  MINERU_FOUND="$(resolve_mineru_cmd)"
  if [[ -n "${MINERU_FOUND}" ]]; then
    export GLOSSAPI_MINERU_COMMAND="${MINERU_FOUND}"
  else
    warn "magic-pdf not found on PATH. Install MinerU or pass --mineru-command to set GLOSSAPI_MINERU_COMMAND."
  fi

  if [[ "${DOWNLOAD_MINERU_MODELS}" -eq 1 ]]; then
    download_mineru_models "${SCRIPT_DIR}/mineru_models"
  else
    warn "MinerU models not downloaded (use --download-mineru-models to fetch the PDF-Extract-Kit bundle)."
  fi
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  pytest_args=("-q")
  case "${MODE}" in
    vanilla)
      pytest_args+=("-m" "not rapidocr and not deepseek")
      ;;
    rapidocr)
      pytest_args+=("-m" "not deepseek")
      ;;
    deepseek)
      pytest_args+=("-m" "not rapidocr")
      ;;
  esac

  info "Running pytest ${pytest_args[*]} tests"
  python_run -m pytest "${pytest_args[@]}" tests
fi

if [[ "${MODE}" == "deepseek" && "${RUN_SMOKE}" -eq 1 ]]; then
  info "Running DeepSeek smoke test"
  python_run "${SCRIPT_DIR}/deepseek_gpu_smoke.py"
fi

cat <<EOF

Environment ready (${MODE}).
Activate with:
  source "${VENV_PATH}/bin/activate"

EOF

if [[ "${MODE}" == "deepseek" ]]; then
  cat <<EOF
DeepSeek-specific exports (add to your shell before running glossapi):
  export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT="${DEEPSEEK_ROOT}/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH="${DEEPSEEK_ROOT}/libjpeg-turbo/lib"
  export GLOSSAPI_DEEPSEEK_ALLOW_STUB=0
  export LD_LIBRARY_PATH="\$GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH:\${LD_LIBRARY_PATH:-}"
EOF
  ENV_FILE="${SCRIPT_DIR}/.env_deepseek"
  cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT="${DEEPSEEK_ROOT}/run_pdf_ocr_vllm.py"
export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH="${DEEPSEEK_ROOT}/libjpeg-turbo/lib"
export GLOSSAPI_DEEPSEEK_ALLOW_STUB=0
export GLOSSAPI_DEEPSEEK_ALLOW_CLI=1
export LD_LIBRARY_PATH="\$GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH:\${LD_LIBRARY_PATH:-}"
EOF
  info "Wrote DeepSeek env exports to ${ENV_FILE} (source it before running OCR)."
fi

if [[ "${MODE}" == "mineru" ]]; then
  MINERU_CONFIG_PATH="${SCRIPT_DIR}/magic-pdf.json"
  MINERU_MODELS_DIR="${SCRIPT_DIR}/mineru_models"
  mkdir -p "${MINERU_MODELS_DIR}"
  MINERU_MODEL_ROOT="${MINERU_MODELS_DIR}"
  if [[ -d "${MINERU_MODELS_DIR}/models" ]]; then
    MINERU_MODEL_ROOT="${MINERU_MODELS_DIR}/models"
  fi
  cat <<EOF > "${MINERU_CONFIG_PATH}"
{
  "bucket_info": {
    "[default]": ["", "", ""]
  },
  "models-dir": "${MINERU_MODEL_ROOT}",
  "device-mode": "cpu",
  "layout-config": {
    "model": "doclayout_yolo"
  },
  "formula-config": {
    "mfd_model": "yolo_v8_mfd",
    "mfr_model": "unimernet_small",
    "enable": true
  }
}
EOF
  cat <<EOF
MinerU-specific exports (optional):
  export GLOSSAPI_MINERU_ALLOW_CLI=1
  export GLOSSAPI_MINERU_ALLOW_STUB=0
  export GLOSSAPI_MINERU_COMMAND="${GLOSSAPI_MINERU_COMMAND:-}"
  export GLOSSAPI_MINERU_MODE="auto"
  export MINERU_TOOLS_CONFIG_JSON="${MINERU_CONFIG_PATH}"
EOF
  ENV_FILE="${SCRIPT_DIR}/.env_mineru"
  cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_MINERU_ALLOW_CLI=1
export GLOSSAPI_MINERU_ALLOW_STUB=0
export GLOSSAPI_MINERU_COMMAND="${GLOSSAPI_MINERU_COMMAND:-}"
export GLOSSAPI_MINERU_MODE="auto"
export MINERU_TOOLS_CONFIG_JSON="${MINERU_CONFIG_PATH}"
EOF
  info "Wrote MinerU env exports to ${ENV_FILE} (source it before running OCR)."
fi

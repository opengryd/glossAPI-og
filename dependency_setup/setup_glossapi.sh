#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="vanilla"
PYTHON_BIN="${PYTHON:-}"
VENV_PATH="${GLOSSAPI_VENV:-}"
DOWNLOAD_DEEPSEEK_OCR=0
DOWNLOAD_DEEPSEEK_OCR2=0
DOWNLOAD_OLMOCR=0
DOWNLOAD_OLMOCR_CUDA=0
DOWNLOAD_GLMOCR=0
GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT:-${REPO_ROOT}/model_weights}"
DEEPSEEK_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/deepseek-ocr"
DEEPSEEK1_MLX_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/deepseek-ocr-1-mlx"
DEEPSEEK2_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/deepseek-ocr-mlx"
OLMOCR_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/olmocr-mlx"
OLMOCR_CUDA_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/olmocr"
GLMOCR_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/glm-ocr-mlx"
MINERU_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/mineru"
MINERU_COMMAND="${GLOSSAPI_MINERU_COMMAND:-}"
DOWNLOAD_MINERU_MODELS=0
MINERU_MODELS_REPO="${MINERU_MODELS_REPO:-opendatalab/PDF-Extract-Kit-1.0}"
RUN_TESTS=0
RUN_SMOKE=0

usage() {
  cat <<'EOF'
Usage: setup_glossapi.sh [options]

Options:
  --mode MODE            Environment profile: vanilla, rapidocr, deepseek-ocr, deepseek-ocr-2, glm-ocr, olmocr, mineru (default: vanilla)
  --venv PATH            Target virtual environment path
  --python PATH          Python executable to use when creating the venv
  --weights-root PATH    Root directory for all model weights (default: $REPO_ROOT/model_weights)
  --download-deepseek-ocr
                         Fetch DeepSeek-OCR weights (only meaningful for --mode deepseek-ocr)
  --download-deepseek-ocr2
                         Fetch DeepSeek OCR v2 weights (only meaningful for --mode deepseek-ocr-2)
  --download-olmocr      Fetch OlmOCR-2 MLX weights (only meaningful for --mode olmocr on macOS)
  --download-olmocr-cuda Fetch OlmOCR-2 CUDA/FP8 weights (only meaningful for --mode olmocr on Linux)
  --download-glmocr      Fetch GLM-OCR MLX weights (only meaningful for --mode glm-ocr)
  --download-mineru-models
                         Download MinerU model bundle
  --mineru-command PATH  Path to magic-pdf binary (optional; stored in GLOSSAPI_MINERU_COMMAND)
  --run-tests            Run pytest -q after installation
  --smoke-test           Run dependency_setup/deepseek_ocr_gpu_smoke.py (deepseek-ocr mode only)
  --help                 Show this help message
EOF
}

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

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
    --download-deepseek-ocr)
      DOWNLOAD_DEEPSEEK_OCR=1
      ;;
    --download-deepseek-ocr2)
      DOWNLOAD_DEEPSEEK_OCR2=1
      ;;
    --weights-root)
      shift || { echo "--weights-root requires a path" >&2; exit 1; }
      GLOSSAPI_WEIGHTS_ROOT="${1:-}"
      DEEPSEEK_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/deepseek-ocr"
      DEEPSEEK1_MLX_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/deepseek-ocr-1-mlx"
      DEEPSEEK2_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/deepseek-ocr-mlx"
      OLMOCR_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/olmocr-mlx"
      OLMOCR_CUDA_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/olmocr"
      GLMOCR_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/glm-ocr-mlx"
      MINERU_WEIGHTS_DIR="${GLOSSAPI_WEIGHTS_ROOT}/mineru"
      ;;
    --download-olmocr)
      DOWNLOAD_OLMOCR=1
      ;;
    --download-olmocr-cuda)
      DOWNLOAD_OLMOCR_CUDA=1
      ;;
    --download-glmocr)
      DOWNLOAD_GLMOCR=1
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
  vanilla|rapidocr|deepseek-ocr|deepseek-ocr-2|glm-ocr|olmocr|mineru) ;;
  *)
    echo "Invalid mode '${MODE}'. Expected vanilla, rapidocr, deepseek-ocr, deepseek-ocr-2, glm-ocr, olmocr, or mineru." >&2
    exit 1
    ;;
esac

REQUIREMENTS_FILE="${SCRIPT_DIR}/base/requirements-glossapi-${MODE}.txt"
if [[ "${MODE}" == "vanilla" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-vanilla-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  fi
fi
if [[ "${MODE}" == "rapidocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-rapidocr-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  fi
fi

if [[ "${MODE}" == "deepseek-ocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-deepseek-ocr-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  fi
fi

if [[ "${MODE}" == "deepseek-ocr-2" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-deepseek-ocr-2-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  else
    warn "deepseek-ocr-2 is macOS-only; falling back to vanilla requirements."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/base/requirements-glossapi-deepseek-ocr-2.txt"
  fi
fi

if [[ "${MODE}" == "olmocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-olmocr-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  fi
fi

if [[ "${MODE}" == "glm-ocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-glm-ocr-macos.txt"
    if [[ -f "${MAC_REQUIREMENTS_FILE}" ]]; then
      REQUIREMENTS_FILE="${MAC_REQUIREMENTS_FILE}"
    fi
  else
    warn "glm-ocr is macOS-only; falling back to vanilla requirements."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/base/requirements-glossapi-glm-ocr.txt"
  fi
fi

if [[ "${MODE}" == "mineru" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    MAC_REQUIREMENTS_FILE="${SCRIPT_DIR}/macos/requirements-glossapi-mineru-macos.txt"
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

python_version_minor() {
  "$1" - <<'PY' 2>/dev/null
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

python_version_tag() {
  local version
  version="$(python_version_minor "$1")"
  if [[ -z "${version}" ]]; then
    echo ""
    return 0
  fi
  echo "${version//./}"
}

if [[ -z "${VENV_PATH}" ]]; then
  py_tag="$(python_version_tag "${PYTHON_BIN}")"
  if [[ -n "${py_tag}" ]]; then
    VENV_PATH="${REPO_ROOT}/dependency_setup/.venvs/${MODE}-py${py_tag}"
  else
    VENV_PATH="${REPO_ROOT}/dependency_setup/.venvs/${MODE}"
  fi
fi

python_supports_glossapi() {
  local version
  version="$(python_version_minor "$1")"
  [[ -z "${version}" ]] && return 1
  local major="${version%%.*}"
  local minor="${version##*.}"
  [[ "${major}" == "3" && "${minor}" -ge 11 && "${minor}" -lt 14 ]]
}

select_glossapi_python() {
  local candidates=(python3.11 python3.12 python3.13)
  local cmd
  for cmd in "${candidates[@]}"; do
    if command -v "${cmd}" >/dev/null 2>&1; then
      if python_supports_glossapi "${cmd}"; then
        echo "${cmd}"
        return 0
      fi
    fi
  done
  echo ""
  return 0
}

ensure_venv() {
  local recreate=0
  if [[ -d "${VENV_PATH}" && ! -x "${VENV_PATH}/bin/python" ]]; then
    warn "Existing venv is missing its Python interpreter. Recreating it."
    recreate=1
  fi

  if [[ -d "${VENV_PATH}" && -x "${VENV_PATH}/bin/python" ]]; then
    local venv_version
    venv_version="$(${VENV_PATH}/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    if ! python_supports_glossapi "${VENV_PATH}/bin/python"; then
      warn "Existing venv uses Python ${venv_version}, which is not supported (requires 3.11–3.13, recommended 3.12). Recreating it."
      recreate=1
    fi
    local requested_version
    requested_version="$(python_version_minor "${PYTHON_BIN}")"
    if [[ -n "${requested_version}" && "${requested_version}" != "${venv_version}" ]]; then
      warn "Existing venv uses Python ${venv_version}, but ${PYTHON_BIN} is ${requested_version}. Recreating it."
      recreate=1
    fi
  fi

  if [[ "${recreate}" -eq 1 ]]; then
    rm -rf "${VENV_PATH}"
  fi

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
import sys
import importlib.util

def _candidate_paths() -> list[Path]:
  candidates: list[Path] = []

  spec = importlib.util.find_spec(
    "magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorchocr.utils.resources"
  )
  if spec and spec.origin:
    candidates.append(Path(spec.origin).parent / "models_config.yml")

  spec = importlib.util.find_spec("mineru.model.utils.pytorchocr.utils.resources")
  if spec and spec.origin:
    candidates.append(Path(spec.origin).parent / "models_config.yml")

  for base in map(Path, sys.path):
    candidates.append(
      base
      / "magic_pdf/model/sub_modules/ocr/paddleocr2pytorch/pytorchocr/utils/resources/models_config.yml"
    )
    candidates.append(
      base / "mineru/model/utils/pytorchocr/utils/resources/models_config.yml"
    )

  seen = set()
  ordered: list[Path] = []
  for path in candidates:
    if path in seen:
      continue
    seen.add(path)
    ordered.append(path)
  return ordered


patched = 0
for path in _candidate_paths():
  if not path.exists():
    continue
  text = path.read_text(encoding="utf-8")
  updated = text.replace("det: ch_PP-OCRv3_det_infer.pth", "det: ch_PP-OCRv5_det_infer.pth")
  updated = updated.replace("rec: ch_PP-OCRv4_rec_infer.pth", "rec: ch_PP-OCRv5_rec_infer.pth")
  if updated != text:
    path.write_text(updated, encoding="utf-8")
    print(f"[info] Patched PaddleOCR model config in {path}")
    patched += 1

if patched == 0:
  print("[warn] PaddleOCR models_config.yml not found for patching.")
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

# ── Shared HuggingFace model downloader ──────────────────────────────────
# Usage: download_hf_model <label> <repo_id> <target_dir> <rerun_hint>
# Tries: 1) hf CLI  2) git-lfs clone  3) warns
download_hf_model() {
  local label="$1"
  local repo_id="$2"
  local target="$3"
  local rerun_hint="$4"

  if [[ -d "${target}" && -f "${target}/config.json" ]]; then
    info "${label} weights already present at ${target}"
    return 0
  fi

  mkdir -p "${target}"
  local hf_cli=""
  if [[ -x "${VENV_PATH}/bin/hf" ]]; then
    hf_cli="${VENV_PATH}/bin/hf"
  elif command -v hf >/dev/null 2>&1; then
    hf_cli="hf"
  fi
  if [[ -n "${hf_cli}" ]]; then
    info "Downloading ${label} weights with hf CLI (this may take a while)"
    "${hf_cli}" download "${repo_id}" \
      --repo-type model \
      --local-dir "${target}" || warn "hf download failed; falling back to git-lfs"
  fi

  if [[ ! -f "${target}/config.json" ]]; then
    if command -v git >/dev/null 2>&1; then
      if ! command -v git-lfs >/dev/null 2>&1; then
        warn "git-lfs not available; install git-lfs to clone ${label} weights via git."
      else
        info "Cloning ${label} weights via git-lfs"
        git lfs install --skip-repo >/dev/null 2>&1 || true
        git clone "https://huggingface.co/${repo_id}" "${target}"
      fi
    else
      warn "Neither hf CLI nor git found; skipping ${label} weight download."
    fi
  fi

  if [[ ! -f "${target}/config.json" ]]; then
    warn "${label} weights were not downloaded. Set GLOSSAPI_WEIGHTS_ROOT and re-run with ${rerun_hint}."
  fi
}

download_deepseek_ocr_weights() {
  download_hf_model "DeepSeek-OCR" "deepseek-ai/DeepSeek-OCR" "$1" "--download-deepseek-ocr"
}

download_deepseek_ocr1_mlx_weights() {
  download_hf_model "DeepSeek-OCR v1 MLX" "mlx-community/DeepSeek-OCR-8bit" "$1" "--download-deepseek-ocr"
}

download_deepseek_ocr2_weights() {
  download_hf_model "DeepSeek OCR v2" "mlx-community/DeepSeek-OCR-2-8bit" "$1" "--download-deepseek-ocr2"
}

download_olmocr_weights() {
  download_hf_model "OlmOCR-2 MLX" "mlx-community/olmOCR-2-7B-1025-4bit" "$1" "--download-olmocr"
}

download_olmocr_cuda_weights() {
  download_hf_model "OlmOCR-2 CUDA (FP8)" "allenai/olmOCR-2-7B-1025-FP8" "$1" "--download-olmocr-cuda"
}

download_glmocr_weights() {
  download_hf_model "GLM-OCR MLX" "mlx-community/GLM-OCR-4bit" "$1" "--download-glmocr"
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

if ! python_supports_glossapi "${PYTHON_BIN}"; then
  ALT_PYTHON="$(select_glossapi_python)"
  if [[ -n "${ALT_PYTHON}" ]]; then
    warn "Python ${PYTHON_BIN} is not supported (requires 3.11–3.13, recommended 3.12). Using ${ALT_PYTHON} instead."
    PYTHON_BIN="${ALT_PYTHON}"
  else
    error "Python ${PYTHON_BIN} is not supported (requires 3.11–3.13, recommended 3.12). Install Python 3.11–3.13 and re-run with --python PATH."
  fi
fi

ensure_venv
info "Upgrading pip tooling"
pip_run install --upgrade pip wheel setuptools

info "Installing ${MODE} requirements from $(basename "${REQUIREMENTS_FILE}")"
if [[ "${MODE}" == "mineru" ]]; then
  TMP_REQ="$(mktemp)"
  grep -vE '^(magic-pdf|mineru(\[.*\])?==|-r|--requirement)' "${REQUIREMENTS_FILE}" > "${TMP_REQ}"
  pip_run install -r "${TMP_REQ}"
  rm -f "${TMP_REQ}"
  info "Installing mineru without dependencies to avoid deep resolver"
  pip_run install --no-deps "mineru[all]==2.7.5"
  info "Installing magic-pdf without dependencies"
  pip_run install --no-deps "magic-pdf==1.3.12"
  info "Installing magic-pdf runtime deps"
  pip_run install "loguru"
elif [[ "${MODE}" == "glm-ocr" ]]; then
  # The GLM-OCR requirements file is self-contained (no -r vanilla) and already
  # pins transformers>=5.1, huggingface-hub>=1.0 — no docling lines.  Install
  # it first, then add docling packages with --no-deps so their transitive
  # transformers<5 / huggingface_hub<1 constraints cannot trigger downgrades.
  pip_run install -r "${REQUIREMENTS_FILE}"
  info "Installing Docling packages (--no-deps to avoid transformers<5 constraint)"
  pip_run install --no-deps docling==2.48.0 docling-core==2.47.0 \
    docling-parse==4.4.0 docling-ibm-models==3.9.1 \
    || warn "Docling install failed; Docling-based extraction will be unavailable."
  info "Installing mlx-lm and mlx-vlm (--no-deps)"
  pip_run install --no-deps "mlx-lm>=0.30.7"   || warn "mlx-lm upgrade failed"
  pip_run install --no-deps "mlx-vlm>=0.3.12"  || warn "mlx-vlm install failed; GLM-OCR in-process MLX mode will be unavailable."
else
  pip_run install -r "${REQUIREMENTS_FILE}"
fi

info "Installing glossapi in editable mode"
pip_run install -e "${REPO_ROOT}" --no-deps

info "Building Rust extensions via editable installs"
pip_run install -e "${REPO_ROOT}/rust/glossapi_rs_cleaner"
pip_run install -e "${REPO_ROOT}/rust/glossapi_rs_noise"

if [[ "${MODE}" == "deepseek-ocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    info "Installing mlx-lm and mlx-vlm (--no-deps) for DeepSeek OCR v1 MPS path"
    pip_run install --no-deps "mlx-lm>=0.30.7"  || warn "mlx-lm upgrade failed"
    pip_run install --no-deps "mlx-vlm>=0.3.12" || warn "mlx-vlm install failed; DeepSeek OCR v1 in-process MLX mode will be unavailable."
    export GLOSSAPI_DEEPSEEK_OCR_DEVICE="mps"
    export GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0
    export GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON="${VENV_PATH}/bin/python"

    if [[ "${DOWNLOAD_DEEPSEEK_OCR}" -eq 1 ]]; then
      download_deepseek_ocr1_mlx_weights "${DEEPSEEK1_MLX_WEIGHTS_DIR}"
      if [[ -d "${DEEPSEEK1_MLX_WEIGHTS_DIR}" && -f "${DEEPSEEK1_MLX_WEIGHTS_DIR}/config.json" ]]; then
        export GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR="${DEEPSEEK1_MLX_WEIGHTS_DIR}"
        info "DeepSeek OCR v1 MLX model dir set to ${GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR}"
      fi
    else
      info "DeepSeek OCR v1 MLX weights not pre-downloaded; model will auto-download from HuggingFace at first run."
    fi
  else
    export GLOSSAPI_DEEPSEEK_OCR_PYTHON="${VENV_PATH}/bin/python"
    export GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT="${DEEPSEEK_WEIGHTS_DIR}/run_pdf_ocr_vllm.py"
    export GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH="${DEEPSEEK_WEIGHTS_DIR}/libjpeg-turbo/lib"
    export GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0
    export LD_LIBRARY_PATH="${GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

    if [[ "${DOWNLOAD_DEEPSEEK_OCR}" -eq 1 ]]; then
      download_deepseek_ocr_weights "${DEEPSEEK_WEIGHTS_DIR}"
    else
      warn "DeepSeek OCR weights not downloaded (use --download-deepseek-ocr to fetch automatically)."
    fi
  fi
fi

if [[ "${MODE}" == "deepseek-ocr-2" ]]; then
  info "Installing mlx-vlm without dependencies to avoid transformers conflicts"
  pip_run install --no-deps "mlx-vlm>=0.3.12" || warn "mlx-vlm install failed; DeepSeek OCR v2 in-process mode will be unavailable."
  # The MLX CLI script is now shipped inside the glossapi package.
  # GLOSSAPI_DEEPSEEK2_MLX_SCRIPT is only needed to override to an external script.
  export GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0
  export GLOSSAPI_DEEPSEEK2_DEVICE="mps"

  if [[ "${DOWNLOAD_DEEPSEEK_OCR2}" -eq 1 ]]; then
    download_deepseek_ocr2_weights "${DEEPSEEK2_WEIGHTS_DIR}"
    if [[ -d "${DEEPSEEK2_WEIGHTS_DIR}" && -f "${DEEPSEEK2_WEIGHTS_DIR}/config.json" ]]; then
      export GLOSSAPI_DEEPSEEK2_MODEL_DIR="${DEEPSEEK2_WEIGHTS_DIR}"
      info "DeepSeek OCR v2 model dir set to ${GLOSSAPI_DEEPSEEK2_MODEL_DIR}"
    fi
  else
    info "DeepSeek OCR v2 weights not pre-downloaded; model will auto-download from HuggingFace at first run."
  fi
fi

# ── Helper: locate the CUDA runtime library directory ───────────────────
# Tries CUDA_HOME, common system paths, and ldconfig in that order.
_detect_cuda_lib_path() {
  # 1. CUDA_HOME / CUDA_PATH
  for root in "${CUDA_HOME:-}" "${CUDA_PATH:-}"; do
    if [[ -n "${root}" ]]; then
      for sub in lib64 lib; do
        if [[ -f "${root}/${sub}/libcudart.so" || -f "${root}/${sub}/libcudart.so.12" ]]; then
          echo "${root}/${sub}"
          return 0
        fi
      done
    fi
  done
  # 2. Well-known system paths
  for candidate in /usr/local/cuda/lib64 /usr/local/cuda/lib \
                   /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64; do
    if [[ -f "${candidate}/libcudart.so" || -f "${candidate}/libcudart.so.12" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  # 3. ldconfig
  if command -v ldconfig >/dev/null 2>&1; then
    local libpath
    libpath="$(ldconfig -p 2>/dev/null | grep -m1 'libcudart\.so' | sed 's/.*=> //' | xargs -r dirname 2>/dev/null)"
    if [[ -n "${libpath}" ]]; then
      echo "${libpath}"
      return 0
    fi
  fi
  echo ""
}

if [[ "${MODE}" == "olmocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    info "Installing mlx-vlm without dependencies to avoid transformers conflicts"
    pip_run install --no-deps "mlx-vlm>=0.3.12" || warn "mlx-vlm install failed; OlmOCR in-process MLX mode will be unavailable."
    export GLOSSAPI_OLMOCR_ENABLE_STUB=0
    export GLOSSAPI_OLMOCR_ENABLE_OCR=1
    export GLOSSAPI_OLMOCR_DEVICE="mps"
  else
    # ── Linux / CUDA setup ──────────────────────────────────────────────
    # The requirements step (vanilla deps via docling-ibm-models) may have
    # already installed a CPU-only torch from PyPI.  We must replace it
    # with a CUDA-enabled build from the PyTorch wheel index.
    #
    # Architecture handling:
    #   x86_64  → cu124 index (standard CUDA 12.4 wheels)
    #   aarch64 → cu126 index (ARM64 CUDA wheels; cu124 only has x86_64)
    _torch_index="https://download.pytorch.org/whl/cu124"
    if [[ "$(uname -m)" == "aarch64" ]]; then
      _torch_index="https://download.pytorch.org/whl/cu126"
      info "Detected aarch64 architecture — using cu126 wheel index for CUDA PyTorch"
    fi

    # 1. Uninstall any existing CPU-only torch, then install CUDA-enabled
    #    torch from the correct index.  Using uninstall+install instead of
    #    --force-reinstall avoids reinstalling all transitive deps.
    info "OlmOCR on Linux/CUDA: installing CUDA-enabled PyTorch"
    pip_run uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip_run install torch torchvision torchaudio \
      --index-url "${_torch_index}" \
      || warn "CUDA PyTorch install failed; falling back to default torch (may be CPU-only)."

    # Verify CUDA is actually available in the new torch install.
    if python_run -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      info "CUDA PyTorch verified — torch.cuda.is_available() == True"
    else
      warn "torch.cuda.is_available() reports False after install."
      warn "The installed torch may be CPU-only (architecture: $(uname -m))."
      warn "Attempting fallback: installing torch from PyTorch nightly index…"
      pip_run uninstall -y torch torchvision torchaudio 2>/dev/null || true
      pip_run install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/nightly/cu126" \
        || warn "Nightly CUDA PyTorch install also failed."
      if python_run -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        info "CUDA PyTorch verified via nightly — torch.cuda.is_available() == True"
      else
        warn "CUDA is still not available. OCR will fail unless you install CUDA PyTorch manually."
        warn "Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
      fi
    fi

    # 2. Install vLLM for the built-in in-process/CLI CUDA runner.
    #    Use --no-deps to prevent vLLM from pulling in a CPU-only torch
    #    that would overwrite the CUDA build we just installed.
    info "Installing vLLM for in-process CUDA execution"
    pip_run install --no-deps "vllm>=0.6.0" \
      || warn "vLLM install failed; in-process CUDA mode will be unavailable."
    # Install vLLM's remaining dependencies (excluding torch).
    pip_run install "vllm>=0.6.0" --no-build-isolation 2>/dev/null \
      || true  # deps may already be satisfied from requirements step

    # 3. Install olmocr[gpu] as fallback pipeline runner.
    #    Again avoid letting it overwrite our CUDA torch.
    info "Installing olmocr[gpu] as fallback CLI runner"
    pip_run install --no-deps "olmocr[gpu]" \
      || warn "olmocr[gpu] install failed; OlmOCR CLI fallback will be unavailable."
    pip_run install "olmocr[gpu]" --no-build-isolation 2>/dev/null \
      || true  # deps may already be satisfied

    # 4. Final CUDA verification — make sure vLLM/olmocr didn't clobber torch.
    if python_run -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      info "Final CUDA check passed — torch.cuda.is_available() == True"
    else
      warn "CUDA torch was overwritten during vLLM/olmocr install — reinstalling…"
      pip_run uninstall -y torch torchvision torchaudio 2>/dev/null || true
      pip_run install torch torchvision torchaudio \
        --index-url "${_torch_index}" \
        || warn "CUDA PyTorch re-install failed."
    fi

    # 5. Detect CUDA runtime library path for subprocess LD_LIBRARY_PATH.
    OLMOCR_CUDA_LIB_PATH="$(_detect_cuda_lib_path)"
    if [[ -n "${OLMOCR_CUDA_LIB_PATH}" ]]; then
      info "Detected CUDA runtime libs at ${OLMOCR_CUDA_LIB_PATH}"
      export GLOSSAPI_OLMOCR_LD_LIBRARY_PATH="${OLMOCR_CUDA_LIB_PATH}"
    else
      warn "Could not auto-detect CUDA library path. If you hit 'libcudart.so.12 not found' errors, set GLOSSAPI_OLMOCR_LD_LIBRARY_PATH manually."
    fi

    export GLOSSAPI_OLMOCR_ENABLE_STUB=0
    export GLOSSAPI_OLMOCR_ENABLE_OCR=1
    export GLOSSAPI_OLMOCR_DEVICE="cuda"
  fi

  # Weight downloads: macOS → MLX weights, Linux → CUDA weights.
  # The --download-olmocr flag is platform-aware.
  if [[ "${DOWNLOAD_OLMOCR}" -eq 1 ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      download_olmocr_weights "${OLMOCR_WEIGHTS_DIR}"
      if [[ -d "${OLMOCR_WEIGHTS_DIR}" && -f "${OLMOCR_WEIGHTS_DIR}/config.json" ]]; then
        export GLOSSAPI_OLMOCR_MLX_MODEL_DIR="${OLMOCR_WEIGHTS_DIR}"
        info "OlmOCR MLX model dir set to ${GLOSSAPI_OLMOCR_MLX_MODEL_DIR}"
      fi
    else
      # On Linux, --download-olmocr fetches CUDA/FP8 weights.
      download_olmocr_cuda_weights "${OLMOCR_CUDA_WEIGHTS_DIR}"
      if [[ -d "${OLMOCR_CUDA_WEIGHTS_DIR}" && -f "${OLMOCR_CUDA_WEIGHTS_DIR}/config.json" ]]; then
        export GLOSSAPI_OLMOCR_MODEL_DIR="${OLMOCR_CUDA_WEIGHTS_DIR}"
        info "OlmOCR CUDA model dir set to ${GLOSSAPI_OLMOCR_MODEL_DIR}"
      fi
    fi
  else
    if [[ "$(uname -s)" == "Darwin" ]]; then
      info "OlmOCR MLX weights not pre-downloaded; model will auto-download from HuggingFace at first run."
    else
      info "OlmOCR CUDA weights not pre-downloaded; model will auto-download from HuggingFace at first run."
    fi
  fi

  # Explicit --download-olmocr-cuda always downloads CUDA weights regardless of platform.
  if [[ "${DOWNLOAD_OLMOCR_CUDA}" -eq 1 ]]; then
    download_olmocr_cuda_weights "${OLMOCR_CUDA_WEIGHTS_DIR}"
    if [[ -d "${OLMOCR_CUDA_WEIGHTS_DIR}" && -f "${OLMOCR_CUDA_WEIGHTS_DIR}/config.json" ]]; then
      export GLOSSAPI_OLMOCR_MODEL_DIR="${OLMOCR_CUDA_WEIGHTS_DIR}"
      info "OlmOCR CUDA model dir set to ${GLOSSAPI_OLMOCR_MODEL_DIR}"
    fi
  fi
fi

if [[ "${MODE}" == "glm-ocr" ]]; then
  # Package installs already handled above in the main install block.
  export GLOSSAPI_GLMOCR_ENABLE_STUB=0
  export GLOSSAPI_GLMOCR_ENABLE_OCR=1
  export GLOSSAPI_GLMOCR_DEVICE="mps"

  if [[ "${DOWNLOAD_GLMOCR}" -eq 1 ]]; then
    download_glmocr_weights "${GLMOCR_WEIGHTS_DIR}"
    if [[ -d "${GLMOCR_WEIGHTS_DIR}" && -f "${GLMOCR_WEIGHTS_DIR}/config.json" ]]; then
      export GLOSSAPI_GLMOCR_MODEL_DIR="${GLMOCR_WEIGHTS_DIR}"
      info "GLM-OCR MLX model dir set to ${GLOSSAPI_GLMOCR_MODEL_DIR}"
    fi
  else
    info "GLM-OCR weights not pre-downloaded; model will auto-download from HuggingFace at first run."
  fi
fi

if [[ "${MODE}" == "mineru" ]]; then
  info "Patching Unimernet for transformers cache_position compatibility"
  patch_mineru_unimernet_cache_position || warn "Unimernet patch failed; math formula recognition may error on newer transformers."
  info "Patching Unimernet SDPA cache handling"
  patch_mineru_unimernet_sdpa_none_cache || warn "Unimernet SDPA cache patch failed."
  info "Patching PaddleOCR model mapping"
  patch_mineru_paddleocr_config || warn "PaddleOCR model config patch failed."

  DETECTRON2_AUTO_INSTALL=${DETECTRON2_AUTO_INSTALL:-0}
  if [[ -n "${DETECTRON2_WHL_URL:-}" ]]; then
    info "Installing detectron2 from DETECTRON2_WHL_URL"
    pip_run install "${DETECTRON2_WHL_URL}" || warn "Detectron2 install failed; MinerU CLI may require a source build on macOS."
  elif [[ "${DETECTRON2_AUTO_INSTALL}" == "1" && "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    info "Installing detectron2 from source (macOS arm64)"
    CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" pip_run install --no-build-isolation \
      "git+https://github.com/facebookresearch/detectron2.git" \
      || warn "Detectron2 source install failed; MinerU CLI will use stub fallback unless detectron2 is installed."
  fi

  info "Installing unimernet (no deps) to avoid transformer downgrades"
  pip_run install --no-deps "unimernet" || warn "Unimernet install failed; MinerU math features may be unavailable."

  DETECTRON2_AVAILABLE=0
  if python_run - <<'PY'
import importlib.util as iu
raise SystemExit(0 if iu.find_spec("detectron2") else 1)
PY
  then
    DETECTRON2_AVAILABLE=1
  fi

  MINERU_FOUND="$(resolve_mineru_cmd)"
  if [[ -n "${MINERU_FOUND}" ]]; then
    export GLOSSAPI_MINERU_COMMAND="${MINERU_FOUND}"
  else
    warn "magic-pdf not found on PATH. Install MinerU or pass --mineru-command to set GLOSSAPI_MINERU_COMMAND."
  fi

  if [[ "${DOWNLOAD_MINERU_MODELS}" -eq 1 ]]; then
    download_mineru_models "${MINERU_WEIGHTS_DIR}"
  else
    warn "MinerU models not downloaded (use --download-mineru-models to fetch the PDF-Extract-Kit bundle)."
  fi
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  pytest_args=("-q")
  case "${MODE}" in
    vanilla)
      pytest_args+=("-m" "not rapidocr and not deepseek_ocr")
      ;;
    rapidocr)
      pytest_args+=("-m" "not deepseek_ocr")
      ;;
    deepseek-ocr)
      pytest_args+=("-m" "not rapidocr")
      ;;
    deepseek-ocr-2)
      pytest_args+=("-m" "not rapidocr and not deepseek_ocr")
      ;;
    olmocr)
      pytest_args+=("-m" "not rapidocr and not deepseek_ocr")
      ;;
    glm-ocr)
      pytest_args+=("-m" "not rapidocr and not deepseek_ocr")
      ;;
  esac

  info "Running pytest ${pytest_args[*]} tests"
  python_run -m pytest "${pytest_args[@]}" tests
fi

if [[ "${MODE}" == "deepseek-ocr" && "${RUN_SMOKE}" -eq 1 ]]; then
  info "Running DeepSeek OCR smoke test"
  python_run "${SCRIPT_DIR}/deepseek_ocr_gpu_smoke.py"
fi

cat <<EOF

Environment ready (${MODE}).
Activate with:
  source "${VENV_PATH}/bin/activate"

EOF

if [[ "${MODE}" == "deepseek-ocr" ]]; then
  ENV_FILE="${SCRIPT_DIR}/.env_deepseek_ocr"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    cat <<EOF
DeepSeek OCR v1 (MLX/MPS) exports (add to your shell before running glossapi):
  export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
  export GLOSSAPI_DEEPSEEK_OCR_DEVICE="mps"
  export GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0
EOF
    cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
export GLOSSAPI_DEEPSEEK_OCR_DEVICE="mps"
export GLOSSAPI_DEEPSEEK_OCR_TEST_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0
EOF
    if [[ -n "${GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR:-}" ]]; then
      echo "export GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR=\"${GLOSSAPI_DEEPSEEK_OCR_MLX_MODEL_DIR}\"" >> "${ENV_FILE}"
    fi
  else
    cat <<EOF
DeepSeek OCR exports (add to your shell before running glossapi):
  export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
  export GLOSSAPI_DEEPSEEK_OCR_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT="${DEEPSEEK_WEIGHTS_DIR}/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH="${DEEPSEEK_WEIGHTS_DIR}/libjpeg-turbo/lib"
  export GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0
  export LD_LIBRARY_PATH="\$GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH:\${LD_LIBRARY_PATH:-}"
EOF
    cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
export GLOSSAPI_DEEPSEEK_OCR_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_DEEPSEEK_OCR_VLLM_SCRIPT="${DEEPSEEK_WEIGHTS_DIR}/run_pdf_ocr_vllm.py"
export GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH="${DEEPSEEK_WEIGHTS_DIR}/libjpeg-turbo/lib"
export GLOSSAPI_DEEPSEEK_OCR_ENABLE_STUB=0
export GLOSSAPI_DEEPSEEK_OCR_ENABLE_OCR=1
export LD_LIBRARY_PATH="\$GLOSSAPI_DEEPSEEK_OCR_LD_LIBRARY_PATH:\${LD_LIBRARY_PATH:-}"
EOF
  fi
  info "Wrote DeepSeek OCR env exports to ${ENV_FILE} (source it before running OCR)."
fi

if [[ "${MODE}" == "deepseek-ocr-2" ]]; then
  cat <<EOF
DeepSeek OCR v2 (MLX/MPS) exports (add to your shell before running glossapi):
  export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
  export GLOSSAPI_DEEPSEEK2_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0
  export GLOSSAPI_DEEPSEEK2_ENABLE_OCR=1
  export GLOSSAPI_DEEPSEEK2_DEVICE="mps"
EOF
  ENV_FILE="${SCRIPT_DIR}/.env_deepseek_ocr2"
  cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
export GLOSSAPI_DEEPSEEK2_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_DEEPSEEK2_ENABLE_STUB=0
export GLOSSAPI_DEEPSEEK2_ENABLE_OCR=1
export GLOSSAPI_DEEPSEEK2_DEVICE="mps"
EOF
  info "Wrote DeepSeek OCR v2 env exports to ${ENV_FILE} (source it before running OCR)."
fi

if [[ "${MODE}" == "olmocr" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    cat <<EOF
OlmOCR-2 (MLX/MPS) exports (add to your shell before running glossapi):
  export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
  export GLOSSAPI_OLMOCR_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_OLMOCR_ENABLE_STUB=0
  export GLOSSAPI_OLMOCR_ENABLE_OCR=1
  export GLOSSAPI_OLMOCR_DEVICE="mps"
EOF
    ENV_FILE="${SCRIPT_DIR}/.env_olmocr"
    cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
export GLOSSAPI_OLMOCR_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_OLMOCR_ENABLE_STUB=0
export GLOSSAPI_OLMOCR_ENABLE_OCR=1
export GLOSSAPI_OLMOCR_DEVICE="mps"
EOF
    if [[ -n "${GLOSSAPI_OLMOCR_MLX_MODEL_DIR:-}" ]]; then
      echo "export GLOSSAPI_OLMOCR_MLX_MODEL_DIR=\"${GLOSSAPI_OLMOCR_MLX_MODEL_DIR}\"" >> "${ENV_FILE}"
    fi
  else
    # Build LD_LIBRARY_PATH export line only if we detected a path.
    OLMOCR_LD_LINE=""
    if [[ -n "${GLOSSAPI_OLMOCR_LD_LIBRARY_PATH:-}" ]]; then
      OLMOCR_LD_LINE="  export GLOSSAPI_OLMOCR_LD_LIBRARY_PATH=\"${GLOSSAPI_OLMOCR_LD_LIBRARY_PATH}\""
    fi
    cat <<EOF
OlmOCR-2 (CUDA/vLLM) exports (add to your shell before running glossapi):
  export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
  export GLOSSAPI_OLMOCR_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_OLMOCR_ENABLE_STUB=0
  export GLOSSAPI_OLMOCR_ENABLE_OCR=1
  export GLOSSAPI_OLMOCR_DEVICE="cuda"
EOF
    if [[ -n "${OLMOCR_LD_LINE}" ]]; then
      echo "${OLMOCR_LD_LINE}"
    fi
    ENV_FILE="${SCRIPT_DIR}/.env_olmocr"
    cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
export GLOSSAPI_OLMOCR_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_OLMOCR_ENABLE_STUB=0
export GLOSSAPI_OLMOCR_ENABLE_OCR=1
export GLOSSAPI_OLMOCR_DEVICE="cuda"
EOF
    if [[ -n "${GLOSSAPI_OLMOCR_LD_LIBRARY_PATH:-}" ]]; then
      echo "export GLOSSAPI_OLMOCR_LD_LIBRARY_PATH=\"${GLOSSAPI_OLMOCR_LD_LIBRARY_PATH}\"" >> "${ENV_FILE}"
    fi
    if [[ -n "${GLOSSAPI_OLMOCR_MODEL_DIR:-}" ]]; then
      echo "export GLOSSAPI_OLMOCR_MODEL_DIR=\"${GLOSSAPI_OLMOCR_MODEL_DIR}\"" >> "${ENV_FILE}"
    fi
  fi
  info "Wrote OlmOCR env exports to ${ENV_FILE} (source it before running OCR)."
fi

if [[ "${MODE}" == "glm-ocr" ]]; then
  cat <<EOF
GLM-OCR (MLX/MPS) exports (add to your shell before running glossapi):
  export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
  export GLOSSAPI_GLMOCR_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_GLMOCR_ENABLE_STUB=0
  export GLOSSAPI_GLMOCR_ENABLE_OCR=1
  export GLOSSAPI_GLMOCR_DEVICE="mps"
EOF
  ENV_FILE="${SCRIPT_DIR}/.env_glmocr"
  cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_WEIGHTS_ROOT="${GLOSSAPI_WEIGHTS_ROOT}"
export GLOSSAPI_GLMOCR_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_GLMOCR_ENABLE_STUB=0
export GLOSSAPI_GLMOCR_ENABLE_OCR=1
export GLOSSAPI_GLMOCR_DEVICE="mps"
EOF
  info "Wrote GLM-OCR env exports to ${ENV_FILE} (source it before running OCR)."
fi

if [[ "${MODE}" == "mineru" ]]; then
  MINERU_CONFIG_PATH="${MINERU_WEIGHTS_DIR}/magic-pdf.json"
  mkdir -p "${MINERU_WEIGHTS_DIR}"
  MINERU_MODEL_ROOT="${MINERU_WEIGHTS_DIR}"
  if [[ -d "${MINERU_WEIGHTS_DIR}/models" ]]; then
    MINERU_MODEL_ROOT="${MINERU_WEIGHTS_DIR}/models"
  fi
  MINERU_DEVICE_MODE_DEFAULT="cpu"
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    MINERU_DEVICE_MODE_DEFAULT="mps"
  fi
  cat <<EOF > "${MINERU_CONFIG_PATH}"
{
  "bucket_info": {
    "[default]": ["", "", ""]
  },
  "models-dir": "${MINERU_MODEL_ROOT}",
  "device-mode": "${MINERU_DEVICE_MODE_DEFAULT}",
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
  MINERU_ENABLE_STUB_DEFAULT=0
  if [[ "${DETECTRON2_AVAILABLE}" -eq 0 ]]; then
    warn "detectron2 not available; enabling MinerU stub fallback (set GLOSSAPI_MINERU_ENABLE_STUB=0 after installing detectron2)."
    MINERU_ENABLE_STUB_DEFAULT=1
  fi
  cat <<EOF
MinerU-specific exports (optional):
  export GLOSSAPI_MINERU_ENABLE_OCR=1
  export GLOSSAPI_MINERU_ENABLE_STUB=${MINERU_ENABLE_STUB_DEFAULT}
  export GLOSSAPI_MINERU_COMMAND="${GLOSSAPI_MINERU_COMMAND:-}"
  export GLOSSAPI_MINERU_MODE="auto"
  export GLOSSAPI_SKIP_DOCLING_BOOT=1
  export MINERU_TOOLS_CONFIG_JSON="${MINERU_CONFIG_PATH}"
EOF
  ENV_FILE="${SCRIPT_DIR}/.env_mineru"
  cat <<EOF > "${ENV_FILE}"
export GLOSSAPI_MINERU_ENABLE_OCR=1
export GLOSSAPI_MINERU_ENABLE_STUB=${MINERU_ENABLE_STUB_DEFAULT}
export GLOSSAPI_MINERU_COMMAND="${GLOSSAPI_MINERU_COMMAND:-}"
export GLOSSAPI_MINERU_MODE="auto"
export GLOSSAPI_SKIP_DOCLING_BOOT=1
export MINERU_TOOLS_CONFIG_JSON="${MINERU_CONFIG_PATH}"
EOF
  info "Wrote MinerU env exports to ${ENV_FILE} (source it before running OCR)."
fi

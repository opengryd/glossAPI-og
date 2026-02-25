#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<'EOF'

	 ██████╗  ██╗       ██████╗  ███████╗  ███████╗   █████╗   ██████╗  ██╗      ██████╗ ██╗      ██╗
	██╔════╝  ██║      ██╔═══██╗ ██╔════╝  ██╔════╝  ██╔══██╗  ██╔══██╗ ██║     ██╔════╝ ██║      ██║
	██║  ███╗ ██║      ██║   ██║ ███████╗  ███████╗  ███████║  ██████╔╝ ██║     ██║      ██║      ██║
	██║   ██║ ██║      ██║   ██║ ╚════██║  ╚════██║  ██╔══██║  ██╔═══╝  ██║     ██║      ██║      ██║
	╚██████╔╝ ███████╗ ╚██████╔╝ ███████║  ███████║  ██║  ██║  ██║      ██║     ╚██████╗ ███████╗ ██║
	 ╚═════╝  ╚══════╝  ╚═════╝  ╚══════╝  ╚══════╝  ╚═╝  ╚═╝  ╚═╝      ╚═╝      ╚═════╝ ╚══════╝ ╚═╝

	Welcome to GlossAPI CLI
	Unified setup + pipeline launcher for the academic PDF processing toolkit.

	┌──────────────────┬─────┬──────┬─────────┬────────────────────────────────┐
	│ Backend          │ CPU │ CUDA │ MPS/MLX │ Description                    │
	├──────────────────┼─────┼──────┼─────────┼────────────────────────────────┤
	│ vanilla          │  ✓  │  —   │    —    │ Core pipeline                  │
	│ rapidocr         │  ✓  │  ✓   │    ✓    │ Docling + RapidOCR OCR         │
	│ mineru           │  ✓  │  ✓   │    ✓    │ External magic-pdf CLI         │
	│ deepseek-ocr     │  —  │  ✓   │    ✓    │ DeepSeek-OCR                   │
	│ deepseek-ocr-2   │  —  │  —   │    ✓    │ DeepSeek OCR v2                │
	│ glm-ocr          │  —  │  —   │    ✓    │ GLM-OCR 0.5B VLM               │
	│ olmocr           │  —  │  ✓   │    ✓    │ OlmOCR-2 VLM OCR               │
	└──────────────────┴─────┴──────┴─────────┴────────────────────────────────┘

EOF
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
	if command -v python3.11 >/dev/null 2>&1; then
		PYTHON_BIN="python3.11"
	elif command -v python3.12 >/dev/null 2>&1; then
		PYTHON_BIN="python3.12"
	elif command -v python3.13 >/dev/null 2>&1; then
		PYTHON_BIN="python3.13"
	elif command -v python3 >/dev/null 2>&1; then
		PYTHON_BIN="python3"
	elif command -v python >/dev/null 2>&1; then
		PYTHON_BIN="python"
	else
		echo "Python not found on PATH. Install Python 3 to continue." >&2
		exit 1
	fi
fi

select_bootstrap_python() {
	if command -v python3.11 >/dev/null 2>&1; then
		echo "python3.11"
		return 0
	fi
	if command -v python3.12 >/dev/null 2>&1; then
		echo "python3.12"
		return 0
	fi
	if command -v python3.13 >/dev/null 2>&1; then
		echo "python3.13"
		return 0
	fi
	echo "${PYTHON_BIN}"
}

available_python_versions() {
	local versions=()
	local cmd
	for cmd in python3.11 python3.12 python3.13; do
		if command -v "${cmd}" >/dev/null 2>&1; then
			versions+=("${cmd}")
		fi
	done
	[[ "${#versions[@]}" -gt 0 ]] && printf "%s\n" "${versions[@]}"
}

python_version_tag() {
	local version
	version="$("$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
	if [[ -z "${version}" ]]; then
		echo ""
		return 0
	fi
	echo "${version//./}"
}

tty_write() {
	if [[ -w /dev/tty ]]; then
		printf "%s" "$*" > /dev/tty
	else
		printf "%s" "$*"
	fi
}

tty_read() {
	local line
	if [[ -r /dev/tty ]]; then
		IFS= read -r line < /dev/tty
	else
		IFS= read -r line
	fi
	printf "%s" "${line}"
}

prompt_select() {
	local label="$1"
	local default_idx="$2"
	shift 2
	local options=($@)
	local answer
	tty_write "${label}:\n"
	local idx=1
	for opt in "${options[@]}"; do
		tty_write "  ${idx}) ${opt}\n"
		idx=$((idx + 1))
	done
	while true; do
		tty_write "Select [default ${default_idx}]: "
		answer="$(tty_read)"
		if [[ -z "${answer}" ]]; then
			echo "${options[$((default_idx - 1))]}"
			return 0
		fi
		if [[ "${answer}" =~ ^[0-9]+$ ]] && (( answer >= 1 && answer <= ${#options[@]} )); then
			echo "${options[$((answer - 1))]}"
			return 0
		fi
		tty_write "Invalid choice. Try again.\n"
	done
}

prompt_confirm() {
	local label="$1"
	local default="$2"
	local answer
	local default_char="Y"
	if [[ "${default}" != "1" ]]; then
		default_char="n"
	fi
	while true; do
		tty_write "${label} [Y/n] (default ${default_char}): "
		answer="$(tty_read)"
		answer="$(printf '%s' "${answer}" | tr '[:upper:]' '[:lower:]')"
		if [[ -z "${answer}" ]]; then
			[[ "${default}" == "1" ]] && echo "1" || echo "0"
			return 0
		fi
		case "${answer}" in
			y|yes) echo "1"; return 0 ;;
			n|no) echo "0"; return 0 ;;
		esac
		tty_write "Please answer y or n.\n"
	done
}

prompt_text() {
	local label="$1"
	local default="$2"
	local answer
	tty_write "${label} [default: ${default}]: "
	answer="$(tty_read)"
	if [[ -z "${answer}" ]]; then
		echo "${default}"
	else
		echo "${answer}"
	fi
}

find_available_venvs() {
	local venvs=()
	local dir
	if [[ -d "${ROOT_DIR}/dependency_setup/.venvs" ]]; then
		for dir in "${ROOT_DIR}/dependency_setup/.venvs"/*; do
			if [[ -d "${dir}" && -f "${dir}/bin/activate" ]]; then
				venvs+=("${dir}")
			fi
		done
	fi
	if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
		venvs+=("${ROOT_DIR}/.venv")
	fi
	[[ "${#venvs[@]}" -gt 0 ]] && printf "%s\n" "${venvs[@]}"
}

select_existing_venv() {
	local options=("Run setup (create new)" "$@")
	local selection
	if have_gum; then
		selection="$(gum_select "Choose virtualenv" "${options[@]}")"
	else
		selection="$(prompt_select "Choose virtualenv" 1 "${options[@]}")"
	fi
	if [[ "${selection}" == "Run setup (create new)" ]]; then
		VENV_SETUP_REQUESTED=1
		SELECTED_VENV=""
		return 1
	fi
	SELECTED_VENV="${selection}"
}

normalize_action() {
	local value
	value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
	value="${value%% *}"
	case "${value}" in
		setup|s) echo "setup" ;;
		pipeline|run|r) echo "pipeline" ;;
		both|all|b) echo "both" ;;
		*) echo "" ;;
	esac
}

select_action() {
	local selection=""
	if have_gum; then
		selection="$(gum_select "Choose action" "BOTH (setup + pipeline)" "PIPELINE (run only)" "SETUP (install only)")"
	else
		selection="$(prompt_select "Choose action" 1 "BOTH (setup + pipeline)" "PIPELINE (run only)" "SETUP (install only)")"
	fi
	normalize_action "${selection}"
}

confirm_setup() {
	local label="$1"
	if have_gum; then
		gum_confirm "${label}" 1
		return $?
	fi
	local result
	result="$(prompt_confirm "${label}" 1)"
	[[ "${result}" == "1" ]]
}

have_gum() {
	command -v gum >/dev/null 2>&1
}

gum_select() {
	local header="$1"
	shift
	gum choose --header "${header}" "$@"
}

gum_confirm() {
	local label="$1"
	local default="$2"
	if [[ "${default}" == "1" ]]; then
		gum confirm --default=true "${label}"
	else
		gum confirm --default=false "${label}"
	fi
}

gum_input() {
	local label="$1"
	local default="$2"
	local value
	value="$(gum input --prompt "${label}: " --value "${default}")"
	echo "${value}"
}

run_setup_wizard_interactive() {
	local os_name
	os_name="$(uname -s)"
	# rapidocr auto-detects MPS/CoreML/ANE on macOS (via dispatch.py), so it
	# is the recommended default on Apple Silicon as well as Linux/CUDA systems.
	local default_mode="rapidocr"


	local modes=(
		"vanilla"
		"rapidocr"
		"mineru"
		"deepseek-ocr"
		"deepseek-ocr-2"
		"glm-ocr"
		"olmocr"
	)
	local default_idx=1
	local i=1
	for mode in "${modes[@]}"; do
		if [[ "${mode}" == "${default_mode}"* ]]; then
			default_idx=${i}
			break
		fi
		i=$((i + 1))
	done

	local selected_mode
	selected_mode="$(gum_select "Environment profile" "${modes[@]}")"
	selected_mode="${selected_mode%% *}"

	# Warn if user selects macOS-only MLX profiles on Linux
	if [[ "${os_name}" == "Linux" && ("${selected_mode}" == "deepseek-ocr-2" || "${selected_mode}" == "glm-ocr") ]]; then
		echo ""
		echo "⚠️  WARNING: ${selected_mode} uses MLX which is only available on macOS/Apple Silicon."
		echo "   Consider 'deepseek-ocr' (CUDA/vLLM) or 'olmocr' instead."
		echo ""
		if ! gum_confirm "Continue with ${selected_mode} anyway?" 0; then
			echo "Aborted. Re-run to select a different profile."
			exit 0
		fi
	fi

	local py_versions=()
	while IFS= read -r _line; do
		[[ -n "${_line}" ]] && py_versions+=("${_line}")
	done < <(available_python_versions)
	if [[ "${#py_versions[@]}" -eq 0 ]]; then
		echo "Python 3.11–3.13 not found on PATH. Install one (recommended 3.12) and retry." >&2
		exit 1
	fi
	local selected_python
	local py_choices=()
	local version
	for version in "${py_versions[@]}"; do
		if [[ "${version}" == "python3.12" ]]; then
			py_choices+=("${version} (recommended)")
		else
			py_choices+=("${version}")
		fi
	done
	selected_python="$(gum_select "Python version" "${py_choices[@]}")"
	selected_python="${selected_python%% *}"

	local py_tag
	py_tag="$(python_version_tag "${selected_python}")"
	local default_venv
	if [[ -n "${py_tag}" ]]; then
		default_venv="${ROOT_DIR}/dependency_setup/.venvs/${selected_mode}-py${py_tag}"
	else
		default_venv="${ROOT_DIR}/dependency_setup/.venvs/${selected_mode}"
	fi
	local selected_venv
	selected_venv="$(gum_input "Virtualenv path" "${default_venv}")"

	local download_deepseek_ocr=0
	local download_deepseek_ocr2=0
	local download_glmocr=0
	local download_olmocr=0
	local download_mineru=0
	local run_tests=0
	local smoke_test=0
	local weights_root=""
	local detectron2_auto_install=0
	local detectron2_wheel_url=""

	if [[ "${selected_mode}" == "deepseek-ocr" ]]; then
		gum_confirm "Download DeepSeek OCR weights now?" 0 && download_deepseek_ocr=1 || download_deepseek_ocr=0
	fi

	if [[ "${selected_mode}" == "deepseek-ocr-2" ]]; then
		gum_confirm "Download DeepSeek OCR v2 weights now? (skip to auto-download at runtime)" 0 && download_deepseek_ocr2=1 || download_deepseek_ocr2=0
	fi

	if [[ "${selected_mode}" == "glm-ocr" ]]; then
		gum_confirm "Download GLM-OCR weights now? (skip to auto-download at runtime)" 0 && download_glmocr=1 || download_glmocr=0
	fi

	if [[ "${selected_mode}" == "olmocr" ]]; then
		local olmocr_weight_label="OlmOCR-2 weights"
		if [[ "$(uname -s)" != "Darwin" ]]; then
			olmocr_weight_label="OlmOCR-2 CUDA (FP8) weights"
		else
			olmocr_weight_label="OlmOCR-2 MLX weights"
		fi
		gum_confirm "Download ${olmocr_weight_label} now? (skip to auto-download at runtime)" 0 && download_olmocr=1 || download_olmocr=0
	fi

	# Prompt for a shared weights root when any download is requested
	if [[ "${download_deepseek_ocr}" == "1" || "${download_deepseek_ocr2}" == "1" || "${download_glmocr}" == "1" || "${download_olmocr}" == "1" ]]; then
		weights_root="$(gum_input "Model weights root dir" "${ROOT_DIR}/model_weights")"
	fi

	if [[ "${selected_mode}" == "mineru" ]]; then
		gum_confirm "Download MinerU models now?" 1 && download_mineru=1 || download_mineru=0
		detectron2_wheel_url="$(gum_input "Detectron2 wheel URL (optional)" "")"
		if [[ -z "${detectron2_wheel_url}" ]]; then
			gum_confirm "Attempt detectron2 auto-install from source?" 1 && detectron2_auto_install=1 || detectron2_auto_install=0
		fi
	fi

	gum_confirm "Run tests after setup?" 0 && run_tests=1 || run_tests=0
	if [[ "${selected_mode}" == "deepseek-ocr" ]]; then
		gum_confirm "Run DeepSeek OCR smoke test?" 0 && smoke_test=1 || smoke_test=0
	fi

	local args=("--mode" "${selected_mode}" "--venv" "${selected_venv}" "--python" "${selected_python}")
	if [[ -n "${weights_root}" ]]; then
		args+=("--weights-root" "${weights_root}")
	fi
	if [[ "${download_deepseek_ocr}" == "1" ]]; then
		args+=("--download-deepseek-ocr")
	fi
	if [[ "${download_deepseek_ocr2}" == "1" ]]; then
		args+=("--download-deepseek-ocr2")
	fi
	if [[ "${download_glmocr}" == "1" ]]; then
		args+=("--download-glmocr")
	fi
	if [[ "${download_olmocr}" == "1" ]]; then
		args+=("--download-olmocr")
	fi
	if [[ "${download_mineru}" == "1" ]]; then
		args+=("--download-mineru-models")
	fi
	if [[ "${run_tests}" == "1" ]]; then
		args+=("--run-tests")
	fi
	if [[ "${smoke_test}" == "1" ]]; then
		args+=("--smoke-test")
	fi

	local env_args=()
	if [[ -n "${detectron2_wheel_url}" ]]; then
		env_args+=("DETECTRON2_WHL_URL=${detectron2_wheel_url}")
	elif [[ "${detectron2_auto_install}" == "1" ]]; then
		env_args+=("DETECTRON2_AUTO_INSTALL=1")
	fi

	if [[ "${#env_args[@]}" -gt 0 ]]; then
		env "${env_args[@]}" bash "${ROOT_DIR}/dependency_setup/setup_glossapi.sh" "${args[@]}"
	else
		bash "${ROOT_DIR}/dependency_setup/setup_glossapi.sh" "${args[@]}"
	fi

	MODE="${selected_mode}"
	VENV_DIR="${selected_venv}"
}

ensure_bootstrap_venv() {
	BOOTSTRAP_VENV="${ROOT_DIR}/dependency_setup/.venvs/bootstrap"
	BOOTSTRAP_PYTHON="$(select_bootstrap_python)"
	BOOTSTRAP_VERSION="$("${BOOTSTRAP_PYTHON}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
	BOOTSTRAP_MAJOR="${BOOTSTRAP_VERSION%%.*}"
	BOOTSTRAP_MINOR="${BOOTSTRAP_VERSION##*.}"
	if [[ "${BOOTSTRAP_MAJOR}" != "3" || "${BOOTSTRAP_MINOR}" -lt 11 || "${BOOTSTRAP_MINOR}" -ge 14 ]]; then
		echo "Python ${BOOTSTRAP_VERSION} is not supported (requires 3.11–3.13). Install Python 3.11–3.13 and retry." >&2
		exit 1
	fi
	if [[ ! -x "${BOOTSTRAP_VENV}/bin/python" ]]; then
		"${BOOTSTRAP_PYTHON}" -m venv "${BOOTSTRAP_VENV}" || {
			echo "Failed to create bootstrap venv with ${BOOTSTRAP_PYTHON}." >&2
			exit 1
		}
	else
		BOOTSTRAP_VERSION="$("${BOOTSTRAP_VENV}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
		if [[ "${BOOTSTRAP_VERSION}" == "3.14" ]]; then
			rm -rf "${BOOTSTRAP_VENV}"
			"${BOOTSTRAP_PYTHON}" -m venv "${BOOTSTRAP_VENV}" || {
				echo "Failed to recreate bootstrap venv with ${BOOTSTRAP_PYTHON}." >&2
				exit 1
			}
		fi
	fi
	"${BOOTSTRAP_VENV}/bin/python" -m pip install --upgrade pip >/dev/null
	"${BOOTSTRAP_VENV}/bin/python" -m pip install questionary typer rich >/dev/null
}

run_setup_flow() {
	if [[ -z "${MODE}" && -z "${VENV_DIR}" && -z "${DOWNLOAD_MINERU_MODELS}" && -z "${DOWNLOAD_DEEPSEEK_OCR}" && -z "${DOWNLOAD_DEEPSEEK_OCR2}" && -z "${DOWNLOAD_GLMOCR}" && -z "${DOWNLOAD_OLMOCR}" && -z "${WEIGHTS_ROOT}" && -z "${RUN_TESTS}" && -z "${SMOKE_TEST}" ]]; then
		if ! have_gum; then
			echo "gum is required for the interactive setup. Install it and re-run." >&2
			exit 1
		fi
		ensure_bootstrap_venv
		echo "==> Running glossapi setup wizard"
		if ! run_setup_wizard_interactive; then
			exit 1
		fi
		SETUP_RAN=1
	else
		if [[ -z "${MODE}" ]]; then
			echo "MODE is required when using non-interactive flags." >&2
			exit 1
		fi

		if [[ -z "${VENV_DIR}" ]]; then
			py_tag="$(python_version_tag "${PYTHON_BIN}")"
			if [[ -n "${py_tag}" ]]; then
				VENV_DIR="${ROOT_DIR}/dependency_setup/.venvs/${MODE}-py${py_tag}"
			else
				VENV_DIR="${ROOT_DIR}/dependency_setup/.venvs/${MODE}"
			fi
		fi

		args=("--mode" "${MODE}" "--venv" "${VENV_DIR}")

		if [[ -n "${WEIGHTS_ROOT}" ]]; then
			args+=("--weights-root" "${WEIGHTS_ROOT}")
		fi

		if [[ "${MODE}" == "deepseek-ocr" && "${DOWNLOAD_DEEPSEEK_OCR}" == "1" ]]; then
			args+=("--download-deepseek-ocr")
		fi

		if [[ "${MODE}" == "deepseek-ocr-2" && "${DOWNLOAD_DEEPSEEK_OCR2}" == "1" ]]; then
			args+=("--download-deepseek-ocr2")
		fi

		if [[ "${MODE}" == "glm-ocr" && "${DOWNLOAD_GLMOCR}" == "1" ]]; then
			args+=("--download-glmocr")
		fi

		if [[ "${MODE}" == "olmocr" && "${DOWNLOAD_OLMOCR}" == "1" ]]; then
			args+=("--download-olmocr")
		fi

		if [[ "${MODE}" == "mineru" && "${DOWNLOAD_MINERU_MODELS}" == "1" ]]; then
			args+=("--download-mineru-models")
		fi

		if [[ "${RUN_TESTS}" == "1" ]]; then
			args+=("--run-tests")
		fi

		if [[ "${MODE}" == "deepseek-ocr" && "${SMOKE_TEST}" == "1" ]]; then
			args+=("--smoke-test")
		fi

		bash "${ROOT_DIR}/dependency_setup/setup_glossapi.sh" "${args[@]}"
		SETUP_RAN=1
	fi
}

MODE="${MODE:-}"
VENV_DIR="${VENV_DIR:-}"
DOWNLOAD_MINERU_MODELS="${DOWNLOAD_MINERU_MODELS:-}"
DOWNLOAD_DEEPSEEK_OCR="${DOWNLOAD_DEEPSEEK_OCR:-}"
DOWNLOAD_DEEPSEEK_OCR2="${DOWNLOAD_DEEPSEEK_OCR2:-}"
DOWNLOAD_GLMOCR="${DOWNLOAD_GLMOCR:-}"
DOWNLOAD_OLMOCR="${DOWNLOAD_OLMOCR:-}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${GLOSSAPI_WEIGHTS_ROOT:-}}"
RUN_TESTS="${RUN_TESTS:-}"
SMOKE_TEST="${SMOKE_TEST:-}"
VENV_SETUP_REQUESTED=0
SKIP_SETUP_PROMPT=0

ACTION_RAW="${GLOSSAPI_ACTION:-${ACTION:-}}"
ACTION="$(normalize_action "${ACTION_RAW}")"
if [[ -z "${ACTION}" ]]; then
	ACTION="$(select_action)"
fi

WANT_SETUP=0
WANT_PIPELINE=0
case "${ACTION}" in
	setup)
		WANT_SETUP=1
		WANT_PIPELINE=0
		;;
	pipeline)
		WANT_SETUP=0
		WANT_PIPELINE=1
		;;
	both)
		WANT_SETUP=1
		WANT_PIPELINE=1
		;;
	*)
		echo "Unknown action: ${ACTION_RAW}" >&2
		exit 1
		;;
esac

SETUP_RAN=0
if [[ "${WANT_SETUP}" -eq 1 ]]; then
	run_setup_flow
fi

if [[ "${WANT_PIPELINE}" -eq 1 ]]; then
	if [[ -z "${VENV_DIR}" ]]; then
		if [[ -n "${MODE}" ]]; then
			py_tag="$(python_version_tag "${PYTHON_BIN}")"
			if [[ -n "${py_tag}" ]]; then
				VENV_DIR="${ROOT_DIR}/dependency_setup/.venvs/${MODE}-py${py_tag}"
			else
				VENV_DIR="${ROOT_DIR}/dependency_setup/.venvs/${MODE}"
			fi
		fi
	fi

	if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
		FOUND_VENVS=()
		while IFS= read -r _line; do
			[[ -n "${_line}" ]] && FOUND_VENVS+=("${_line}")
		done < <(find_available_venvs)
		if [[ "${#FOUND_VENVS[@]}" -gt 0 ]]; then
			SELECTED_VENV=""
			select_existing_venv "${FOUND_VENVS[@]}" || true
			if [[ -n "${SELECTED_VENV}" ]]; then
				VENV_DIR="${SELECTED_VENV}"
			fi
		fi
	fi

	if [[ "${VENV_SETUP_REQUESTED}" -eq 1 ]]; then
		VENV_SETUP_REQUESTED=0
		MODE=""
		VENV_DIR=""
		SKIP_SETUP_PROMPT=1
		WANT_SETUP=1
		run_setup_flow
	fi

	if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
		if [[ "${SKIP_SETUP_PROMPT}" -eq 1 ]]; then
			echo "Virtualenv not found at ${VENV_DIR}. Run setup first." >&2
			exit 1
		fi
		if [[ "${SETUP_RAN}" -eq 0 && "${WANT_SETUP}" -eq 0 ]]; then
			if confirm_setup "Virtualenv not found. Run setup now?"; then
				WANT_SETUP=1
				run_setup_flow
			else
				echo "Virtualenv not found at ${VENV_DIR}. Run setup first." >&2
				exit 1
			fi
		else
			echo "Virtualenv not found at ${VENV_DIR}. Run setup first." >&2
			exit 1
		fi
	fi

	# shellcheck disable=SC1090
	source "${VENV_DIR}/bin/activate"

	if [[ "${SETUP_RAN}" -eq 0 && -z "${VIRTUAL_ENV:-}" ]]; then
		echo "[warn] Virtualenv not active. Activate it or run setup." >&2
		exit 1
	fi

	if ! command -v glossapi >/dev/null 2>&1; then
		echo "GlossAPI CLI not found. Run setup first." >&2
		exit 1
	fi

	# ---------------------------------------------------------------------------
	# Infer MODE from the selected venv name when not explicitly provided.
	# Strips the trailing -py<digits> version tag (e.g. mineru-py312 → mineru)
	# and validates against the known backend list so bootstrap/custom venvs
	# don't accidentally set MODE to a nonsense value.
	# ---------------------------------------------------------------------------
	if [[ -z "${MODE}" && -n "${VENV_DIR}" ]]; then
		_venv_basename="$(basename "${VENV_DIR}")"
		_inferred="${_venv_basename%-py[0-9]*}"
		case "${_inferred}" in
			vanilla|rapidocr|mineru|deepseek-ocr|deepseek-ocr-2|glm-ocr|olmocr)
				MODE="${_inferred}" ;;
		esac
	fi

	# ---------------------------------------------------------------------------
	# Source the per-backend env file generated by `glossapi setup`.
	# Mapping: mode → env filename (all files live in dependency_setup/).
	# vanilla and rapidocr have no env file (no extra paths/secrets needed).
	# ---------------------------------------------------------------------------
	_source_backend_env() {
		local _mode="$1"
		local _env_dir="${ROOT_DIR}/dependency_setup"
		local _env_file
		case "${_mode}" in
			deepseek-ocr)   _env_file="${_env_dir}/.env_deepseek_ocr"  ;;
			deepseek-ocr-2) _env_file="${_env_dir}/.env_deepseek_ocr2" ;;
			mineru)         _env_file="${_env_dir}/.env_mineru"         ;;
			glm-ocr)        _env_file="${_env_dir}/.env_glmocr"         ;;
			olmocr)         _env_file="${_env_dir}/.env_olmocr"          ;;
			*)              return 0 ;;  # vanilla / rapidocr / unknown — no env file
		esac
		[[ -f "${_env_file}" ]] || return 0
		if ! bash -n "${_env_file}" >/dev/null 2>&1; then
			echo "[warn] ${_mode} env file has syntax errors (${_env_file}); re-run 'glossapi setup' to regenerate it." >&2
			return 0
		fi
		# shellcheck disable=SC1090
		source "${_env_file}"
	}
	_source_backend_env "${MODE}"

	cd "${ROOT_DIR}"
	exec glossapi pipeline
fi

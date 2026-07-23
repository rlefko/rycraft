#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: RYCRAFT_WORLD_SEED=SEED RYCRAFT_SPAWN=X,Y,Z \
       RYCRAFT_YAW=DEGREES RYCRAFT_PITCH=DEGREES \
       ci/run-v4-capture.sh NAME MODEL_PACK [QUALIFICATION_ROOT]

Runs one deterministic generator v4 frame capture from a disposable /tmp
Application Support root. MODEL_PACK is APFS-cloned on the first run, so Core
ML may update its cloned caches without writing to the installed source pack.
Reuse QUALIFICATION_ROOT for related captures with the same generation identity.
The first cloned launch performs one local SHA-256 audit because clone file
stamps differ; later launches reuse the marker refreshed inside the clone.

Optional environment variables:
  RYCRAFT_CAPTURE_BINARY                 Release binary to launch
  RYCRAFT_CAPTURE_CAMERA                 Capture-only camera X,Y,Z
  RYCRAFT_CAPTURE_FRAME                  Scene frame to capture, default 500
  RYCRAFT_CAPTURE_TIMEOUT_SECONDS        Capture deadline, default 45
  RYCRAFT_CAPTURE_METAL_VALIDATION       Set to 1 for a validation run
  RYCRAFT_VIEW_DISTANCE                  View distance, default 512
  RYCRAFT_TIME                           Absolute world tick, default 6000
  RYCRAFT_WEATHER                        Weather preset, default clear
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

capture_name=$1
source_pack_argument=$2
qualification_root=${3:-}

case "$capture_name" in
    *[!A-Za-z0-9._-]* | '')
        echo "Capture NAME may contain only letters, numbers, periods, underscores, and hyphens" >&2
        exit 2
        ;;
esac

for required_name in RYCRAFT_WORLD_SEED RYCRAFT_SPAWN RYCRAFT_YAW RYCRAFT_PITCH; do
    if [[ -z ${!required_name:-} ]]; then
        echo "$required_name is required for a deterministic capture" >&2
        exit 2
    fi
done

script_directory=$(cd "$(dirname "$0")" && pwd -P)
repository_root=$(cd "$script_directory/.." && pwd -P)
capture_binary=${RYCRAFT_CAPTURE_BINARY:-$repository_root/build-release/src/rycraft}

if [[ ! -x "$capture_binary" ]]; then
    echo "Capture binary is not executable: $capture_binary" >&2
    exit 2
fi
capture_binary=$(cd "$(dirname "$capture_binary")" && pwd -P)/$(basename "$capture_binary")

if [[ ! -d "$source_pack_argument" ]]; then
    echo "Model pack directory does not exist: $source_pack_argument" >&2
    exit 2
fi
source_pack=$(cd "$source_pack_argument" && pwd -P)

model_manifest=$repository_root/resources/config/terrain_model_manifest.json
model_revision=$(/usr/bin/plutil -extract model_revision raw -o - "$model_manifest")
model_assets=()
model_asset_sizes=()
asset_index=0
while asset=$(/usr/bin/plutil -extract "assets.$asset_index.name" raw -o - \
    "$model_manifest" 2>/dev/null); do
    model_assets+=("$asset")
    model_asset_sizes+=("$(/usr/bin/plutil -extract "assets.$asset_index.bytes" raw -o - \
        "$model_manifest")")
    asset_index=$((asset_index + 1))
done
model_assets+=("$(/usr/bin/plutil -extract runtime.name raw -o - "$model_manifest")")
model_asset_sizes+=("$(/usr/bin/plutil -extract runtime.bytes raw -o - "$model_manifest")")

if [[ ! -f "$source_pack/.verified-pack-v1" || -L "$source_pack/.verified-pack-v1" ]]; then
    echo "The source model pack has no verified completion marker" >&2
    exit 3
fi
for asset_index in "${!model_assets[@]}"; do
    asset=${model_assets[$asset_index]}
    expected_size=${model_asset_sizes[$asset_index]}
    if [[ ! -f "$source_pack/$asset" || -L "$source_pack/$asset" ]]; then
        echo "The source model pack is incomplete: missing $asset" >&2
        exit 3
    fi
    actual_size=$(/usr/bin/stat -f '%z' "$source_pack/$asset")
    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "The source model pack has the wrong size for $asset" >&2
        exit 3
    fi
done

if [[ -z "$qualification_root" ]]; then
    qualification_root=$(mktemp -d /tmp/rycraft-v4-qualification.XXXXXX)
else
    if [[ ! -d "$qualification_root" ]]; then
        echo "QUALIFICATION_ROOT must already exist; create it with mktemp -d under /tmp" >&2
        exit 2
    fi
fi
qualification_root=$(cd "$qualification_root" && pwd -P)

case "$qualification_root" in
    /tmp/?* | /private/tmp/?*) ;;
    *)
        echo "QUALIFICATION_ROOT must resolve to a disposable directory under /tmp" >&2
        exit 2
        ;;
esac
case "$qualification_root/" in
    "$repository_root"/*)
        echo "QUALIFICATION_ROOT must remain outside the repository" >&2
        exit 2
        ;;
esac

application_support_root=$qualification_root/application-support
evidence_root=$qualification_root/evidence
runtime_root=$qualification_root/runtime
model_parent=$application_support_root/terrain-models/terrain-diffusion-30m-onnx
cloned_pack=$model_parent/$model_revision

for protected_path in \
    "$application_support_root" \
    "$application_support_root/terrain-models" \
    "$application_support_root/terrain-models/terrain-diffusion-30m-onnx" \
    "$cloned_pack" \
    "$evidence_root" \
    "$runtime_root" \
    "$qualification_root/settings.json"; do
    if [[ -L "$protected_path" ]]; then
        echo "Qualification paths may not contain symlinks: $protected_path" >&2
        exit 2
    fi
done

mkdir -p "$model_parent" "$evidence_root" "$runtime_root"
for protected_path in "$model_parent" "$evidence_root" "$runtime_root"; do
    resolved_path=$(cd "$protected_path" && pwd -P)
    case "$resolved_path/" in
        "$qualification_root"/*) ;;
        *)
            echo "Qualification path escaped its disposable root: $protected_path" >&2
            exit 2
            ;;
    esac
done
capture_path=$evidence_root/$capture_name.png
log_path=$evidence_root/$capture_name.log
capture_frame=${RYCRAFT_CAPTURE_FRAME:-500}
capture_timeout=${RYCRAFT_CAPTURE_TIMEOUT_SECONDS:-45}

if [[ -e "$capture_path" || -e "$log_path" ]]; then
    echo "Capture evidence already exists for NAME: $capture_name" >&2
    exit 2
fi

if [[ ! -d "$cloned_pack" ]]; then
    echo "Cloning the verified model pack into the disposable Application Support root"
    if ! /bin/cp -cR "$source_pack" "$cloned_pack"; then
        echo "The APFS clone failed; no full-copy fallback was attempted" >&2
        exit 3
    fi
    echo "The first cloned launch will audit local files once and will not download them"
fi

for asset_index in "${!model_assets[@]}"; do
    asset=${model_assets[$asset_index]}
    expected_size=${model_asset_sizes[$asset_index]}
    if [[ ! -f "$cloned_pack/$asset" || -L "$cloned_pack/$asset" ]] || \
        [[ $(/usr/bin/stat -f '%z' "$cloned_pack/$asset") != "$expected_size" ]]; then
        echo "The disposable model pack is incomplete: missing $asset" >&2
        exit 3
    fi
done

case "$capture_frame" in
    *[!0-9]* | '')
        echo "RYCRAFT_CAPTURE_FRAME must be an unsigned decimal integer" >&2
        exit 2
        ;;
esac
case "$capture_timeout" in
    *[!0-9]* | '' | 0)
        echo "RYCRAFT_CAPTURE_TIMEOUT_SECONDS must be a positive decimal integer" >&2
        exit 2
        ;;
esac

if [[ ${RYCRAFT_CAPTURE_METAL_VALIDATION:-0} != 0 && ${RYCRAFT_PERF_FRAMES:-0} != 0 ]]; then
    echo "Metal validation and performance capture must run separately" >&2
    exit 2
fi

export RYCRAFT_APPLICATION_SUPPORT_ROOT=$application_support_root
export RYCRAFT_SETTINGS_PATH=$qualification_root/settings.json
export RYCRAFT_CAPTURE=$capture_path
export RYCRAFT_CAPTURE_FRAME=$capture_frame
export RYCRAFT_START_SCREEN=playing
export RYCRAFT_GAME_MODE=${RYCRAFT_GAME_MODE:-creative}
export RYCRAFT_SHOW_DEBUG=${RYCRAFT_SHOW_DEBUG:-1}
export RYCRAFT_GPU_COUNTERS=${RYCRAFT_GPU_COUNTERS:-1}
export RYCRAFT_NATIVE_WINDOW=${RYCRAFT_NATIVE_WINDOW:-1}
export RYCRAFT_TIME=${RYCRAFT_TIME:-6000}
export RYCRAFT_TIME_FREEZE=${RYCRAFT_TIME_FREEZE:-1}
export RYCRAFT_WEATHER=${RYCRAFT_WEATHER:-clear}
export RYCRAFT_VIEW_DISTANCE=${RYCRAFT_VIEW_DISTANCE:-512}
export RYCRAFT_SHADOWS=${RYCRAFT_SHADOWS:-2}
export RYCRAFT_CLOUD_QUALITY=${RYCRAFT_CLOUD_QUALITY:-2}
export RYCRAFT_INDIRECT_LIGHT=${RYCRAFT_INDIRECT_LIGHT:-2}
export RYCRAFT_BLOOM=${RYCRAFT_BLOOM:-1}
export RYCRAFT_VL=${RYCRAFT_VL:-1}
export RYCRAFT_SSR=${RYCRAFT_SSR:-1}
export RYCRAFT_WAVING=${RYCRAFT_WAVING:-1}
export RYCRAFT_LENS_FLARE=${RYCRAFT_LENS_FLARE:-1}
export RYCRAFT_VIBRANCE=${RYCRAFT_VIBRANCE:-5}
export RYCRAFT_SHARPEN=${RYCRAFT_SHARPEN:-0}
unset RYCRAFT_WORLD_DIR

if [[ ${RYCRAFT_CAPTURE_METAL_VALIDATION:-0} != 0 ]]; then
    export MTL_DEBUG_LAYER=1
    export MTL_DEBUG_LAYER_ERROR_MODE=nslog
    export MTL_SHADER_VALIDATION=1
fi

echo "Qualification root: $qualification_root"
echo "Application Support root: $application_support_root"
echo "Capture: $capture_path"
echo "Log: $log_path"

capture_pid=
stop_capture_process() {
    if [[ -n "$capture_pid" ]] && kill -0 "$capture_pid" 2>/dev/null; then
        kill "$capture_pid" 2>/dev/null || true
    fi
}
trap stop_capture_process EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$runtime_root"
"$capture_binary" >"$log_path" 2>&1 &
capture_pid=$!
deadline=$((SECONDS + capture_timeout))

while [[ ! -s "$capture_path" && $SECONDS -lt $deadline ]]; do
    if ! kill -0 "$capture_pid" 2>/dev/null; then
        break
    fi
    sleep 1
done

if [[ ! -s "$capture_path" ]]; then
    if kill -0 "$capture_pid" 2>/dev/null; then
        kill "$capture_pid" 2>/dev/null || true
        wait "$capture_pid" 2>/dev/null || true
        capture_pid=
        echo "Capture did not complete within $capture_timeout seconds" >&2
    else
        set +e
        wait "$capture_pid"
        early_process_status=$?
        set -e
        capture_pid=
        echo "Capture process exited with status $early_process_status before writing a PNG" >&2
    fi
    tail -n 80 "$log_path" >&2 || true
    exit 4
fi

shutdown_deadline=$((SECONDS + 15))
while kill -0 "$capture_pid" 2>/dev/null && [[ $SECONDS -lt $shutdown_deadline ]]; do
    sleep 1
done
if kill -0 "$capture_pid" 2>/dev/null; then
    kill "$capture_pid" 2>/dev/null || true
fi

set +e
wait "$capture_pid"
process_status=$?
set -e
capture_pid=

if [[ $process_status -ne 0 && $process_status -ne 143 ]]; then
    echo "Capture process exited with status $process_status" >&2
    tail -n 80 "$log_path" >&2 || true
    exit "$process_status"
fi

echo "Capture completed"
grep -E \
    'Capture (bootstrap|preparation|evidence|streaming|authority):|Capture tiers \[|Frame captured to|\[ERROR\]|\[FATAL\]|\[MTLDebug\]|validation' \
    "$log_path" || true

if grep -Eq '\[(ERROR|FATAL)\]' "$log_path" || \
    grep -Eiq 'validation error|failed assertion|MTLDebug.*error' "$log_path"; then
    echo "Capture produced generation, runtime, or validation errors" >&2
    exit 5
fi

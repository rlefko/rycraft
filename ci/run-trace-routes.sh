#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ci/run-trace-routes.sh MODEL_PACK [QUALIFICATION_ROOT]

Records the Program 0 critical-path traces for the required routes (cold entry,
warm entry, one-chunk movement, hover, camera reversal, LOD-boundary flight, and
five-minute settlement) from a disposable /tmp Application Support root, then
prints a rycraft_trace_summary for each. MODEL_PACK is APFS-cloned on the first
run so Core ML may update its cloned caches without touching the installed pack.

Each route sets RYCRAFT_TRACE so the game writes ROUTE.json (Chrome trace) and
ROUTE.json.rytrace (binary) at quiesce. Metal counters stay off for trace runs;
collect GPU counters only in the dedicated validation run.

Optional environment variables:
  RYCRAFT_TRACE_BINARY        Debug binary to launch (default build/src/rycraft)
  RYCRAFT_WORLD_SEED          Seed, default 764891
  RYCRAFT_SPAWN               Spawn X,Y,Z, default 23029,225,-111726
  RYCRAFT_VIEW_DISTANCE       View distance, default 512
  RYCRAFT_TRACE_CAPACITY      Trace event capacity, default 4194304
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

source_pack_argument=$1
qualification_root=${2:-}

script_directory=$(cd "$(dirname "$0")" && pwd -P)
repository_root=$(cd "$script_directory/.." && pwd -P)
trace_binary=${RYCRAFT_TRACE_BINARY:-$repository_root/build/src/rycraft}
summary_tool=$repository_root/build/src/rycraft_trace_summary

if [[ ! -x "$trace_binary" ]]; then
    echo "Trace binary is not executable: $trace_binary" >&2
    exit 2
fi
if [[ ! -x "$summary_tool" ]]; then
    echo "Trace summarizer is not built: $summary_tool" >&2
    exit 2
fi
if [[ ! -d "$source_pack_argument" ]]; then
    echo "Model pack directory does not exist: $source_pack_argument" >&2
    exit 2
fi
source_pack=$(cd "$source_pack_argument" && pwd -P)

model_manifest=$repository_root/resources/config/terrain_model_manifest.json
model_revision=$(/usr/bin/plutil -extract model_revision raw -o - "$model_manifest")
if [[ ! -f "$source_pack/.verified-pack-v1" ]]; then
    echo "The source model pack has no verified completion marker" >&2
    exit 3
fi

if [[ -z "$qualification_root" ]]; then
    qualification_root=$(mktemp -d /tmp/rycraft-trace.XXXXXX)
fi
qualification_root=$(cd "$qualification_root" && pwd -P)
case "$qualification_root" in
    /tmp/?* | /private/tmp/?*) ;;
    *)
        echo "QUALIFICATION_ROOT must resolve to a disposable directory under /tmp" >&2
        exit 2
        ;;
esac

application_support_root=$qualification_root/application-support
trace_root=$qualification_root/traces
model_parent=$application_support_root/terrain-models/terrain-diffusion-30m-onnx
cloned_pack=$model_parent/$model_revision
mkdir -p "$model_parent" "$trace_root"

if [[ ! -d "$cloned_pack" ]]; then
    echo "Cloning the verified model pack into the disposable Application Support root"
    /bin/cp -cR "$source_pack" "$cloned_pack"
fi

seed=${RYCRAFT_WORLD_SEED:-764891}
spawn=${RYCRAFT_SPAWN:-23029,225,-111726}
view_distance=${RYCRAFT_VIEW_DISTANCE:-512}
trace_capacity=${RYCRAFT_TRACE_CAPACITY:-4194304}

export RYCRAFT_APPLICATION_SUPPORT_ROOT=$application_support_root
export RYCRAFT_SETTINGS_PATH=$qualification_root/settings.json
export RYCRAFT_WORLD_SEED=$seed
export RYCRAFT_SPAWN=$spawn
export RYCRAFT_YAW=${RYCRAFT_YAW:-0}
export RYCRAFT_PITCH=${RYCRAFT_PITCH:--17}
export RYCRAFT_VIEW_DISTANCE=$view_distance
export RYCRAFT_START_SCREEN=playing
export RYCRAFT_GAME_MODE=${RYCRAFT_GAME_MODE:-creative}
export RYCRAFT_NATIVE_WINDOW=${RYCRAFT_NATIVE_WINDOW:-1}
export RYCRAFT_TRACE_CAPACITY=$trace_capacity
unset RYCRAFT_GPU_COUNTERS
unset RYCRAFT_WORLD_DIR

# route name | RYCRAFT_TRACE_ROUTE | autopilot | warmup | frames | ring
routes=(
    "cold cold '' 0 900 0"
    "warm warm '' 0 900 0"
    "move move walk 300 200 0"
    "hover hover '' 600 300 0"
    "reversal reversal walk 300 400 0"
    "lod lod fly 300 900 0"
    "settle settle '' 600 18000 1"
)

run_route() {
    local name=$1 route=$2 autopilot=$3 warmup=$4 frames=$5 ring=$6
    local trace_path=$trace_root/$name.json
    local log_path=$trace_root/$name.log
    echo "== route $name (route=$route autopilot='${autopilot}' warmup=$warmup frames=$frames ring=$ring)"

    RYCRAFT_TRACE=$trace_path \
    RYCRAFT_TRACE_ROUTE=$route \
    RYCRAFT_TRACE_RING=$ring \
    RYCRAFT_PERF_WARMUP_FRAMES=$warmup \
    RYCRAFT_PERF_FRAMES=$frames \
    ${autopilot:+RYCRAFT_AUTOPILOT=$autopilot} \
        "$trace_binary" >"$log_path" 2>&1 || {
        echo "  route $name exited nonzero; see $log_path" >&2
        tail -n 40 "$log_path" >&2 || true
        return 1
    }

    if [[ ! -s "$trace_path.rytrace" ]]; then
        echo "  route $name produced no binary trace" >&2
        return 1
    fi
    "$summary_tool" "$trace_path.rytrace"
}

status=0
for spec in "${routes[@]}"; do
    # shellcheck disable=SC2086
    eval set -- $spec
    run_route "$1" "$2" "$3" "$4" "$5" "$6" || status=1
    echo
done

echo "Traces and summaries written under $trace_root"
exit "$status"

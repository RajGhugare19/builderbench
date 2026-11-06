#!/usr/bin/env bash
set -euo pipefail

# Move to repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNS_DIR="outputs/"
MODEL_ID="claude-opus-4-6"
AGENT_NAME="reflexion"
WEBSITE_URL="https://www.anthropic.com/news/claude-opus-4-6"

if [[ ! -d "${RUNS_DIR}" ]]; then
	echo "Runs directory not found: ${RUNS_DIR}" >&2
	exit 1
fi

LEVEL_IDS=$(find "${RUNS_DIR}/${AGENT_NAME}/${MODEL_ID}" -maxdepth 1 -type d -name 'cube-[0-9]*-task-*-seed-*' -printf '%f\n' \
	| sed -E 's/-seed-[0-9]+(-timestamp-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9])?$//' \
	| sort -u)

for level_id in ${LEVEL_IDS}; do
	echo "Submitting ${level_id}"
	python submit.py \
		--results_dir "${RUNS_DIR}" \
		--level_id "${level_id}" \
		--model_id "${MODEL_ID}" \
		--agent_name "${AGENT_NAME}" \
		--website_url "${WEBSITE_URL}"
done

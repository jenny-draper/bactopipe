#!/bin/bash

# Medaka Polishing - BACTOPIPE Medaka Polishing Script
# ==========================================================================
#
# Runs Medaka consensus polishing on bacterial assemblies.
# Automatically detects the appropriate Medaka model from QC run output where available.
#
# Author: Jenny Draper
# Date: October 2025
#
# Usage:
#    run_medaka.sh <SAMPLE_ID> <RUNID> <ASSEMBLY> <READS>
#    run_medaka.sh --version
#

set -euo pipefail

VERSION="1.1"

# Handle version argument
[[ "${1:-}" == "--version" ]] && echo "$(basename "$0") ${VERSION}" && exit 0

# Defaults
MEDAKA_ROUNDS=2
DEFAULT_MEDAKA_MODEL="r1041_e82_400bps_sup_v4.3.0"

# Parse arguments
SAMPLE_ID=$1
RUNID=$2
assembly=$3
reads=$4

# Expand glob if needed, take first match
assembly=($assembly); assembly="${assembly[0]}"
[ ! -f "$assembly" ] && { echo "❌ ERROR: Assembly file not found: $assembly"; exit 1; }

# Set up directories
medaka_dir="/data/runs/ont/analysis/${RUNID}/assembly/medaka/${SAMPLE_ID}"
assembly_dir="/data/runs/ont/analysis/${RUNID}/assembly"
final_output="${assembly_dir}/${SAMPLE_ID}.fa"
mkdir -p "$medaka_dir"


# Determine Medaka model for this run. Attempt to get from run QC data, else use default
MEDAKA_MODEL=$(awk -F'\t' '$1=="Basecaller Model String" {print $2}' "/data/runs/ont/qc/${RUNID}/${RUNID}.ont_versions.tsv" 2>/dev/null)
if [ -z "$MEDAKA_MODEL" ] || ! medaka tools list_models 2>/dev/null | grep -q "^${MEDAKA_MODEL}$"; then
    [ -n "$MEDAKA_MODEL" ] && echo "WARNING: model $MEDAKA_MODEL isn't valid, using default: $DEFAULT_MEDAKA_MODEL" >&2
    [ -z "$MEDAKA_MODEL" ] && echo "⚠️ Could not determine Medaka model for run $RUNID, using default: $DEFAULT_MEDAKA_MODEL"
    MEDAKA_MODEL="$DEFAULT_MEDAKA_MODEL"
else
    echo "✅ Found valid Medaka model: $MEDAKA_MODEL"
fi

echo "▶️ Running Medaka polishing (${MEDAKA_ROUNDS} rounds) with model: $MEDAKA_MODEL"

# Run Medaka polishing rounds
current_fa="$assembly"
for round in $(seq 1 $MEDAKA_ROUNDS); do
    round_dir="${medaka_dir}/round${round}"
    output_fa="${medaka_dir}/${SAMPLE_ID}.m${round}.fa"
    
    echo "   Medaka round ${round}"
    medaka_cmd="medaka_consensus -i $reads -d $current_fa -o $round_dir -m $MEDAKA_MODEL -f -t 8 -b 10"
    echo "   Running: $medaka_cmd"
    eval "$medaka_cmd"
    
    if [ -f "${round_dir}/consensus.fasta" ]; then
        cp "${round_dir}/consensus.fasta" "$output_fa"
        current_fa="$output_fa"
        echo "✅ Medaka round ${round} complete"
    else
        echo "❌ Medaka round ${round} produced no consensus"
        exit 1
    fi
done


# Medaka output strips everything but the contig name from contig headers.
# This is a little python script to restore the header details from the unpolished file
# (but add medaka_polish and update the length)

# Get the actual descriptive filename by following the symlink
actual_assembly=$(readlink -f "$assembly")
unpolished_basename=$(basename "$actual_assembly")

# Create polished filename: replace .fa with -polished.fa
# This works regardless of whether the file has -unpolished in the name or not
final_polished="${medaka_dir}/${unpolished_basename%.fa}-polished.fa"

echo "Creating polished assembly: $final_polished"

python3 -c "
import re
# Load original headers by contig name
orig = {line.split()[0][1:]: line.strip() for line in open('$assembly') if line.startswith('>')}

# Process medaka output
seq = open('$current_fa').read()
for block in seq.split('>')[1:]:
    contig, *seqlines = block.splitlines()
    header = orig.get(contig, f'>{contig}')
    new_len = sum(len(line) for line in seqlines)
    header = re.sub(r'len=\d+', f'len={new_len}', header)
    print(f'{header} medaka_polish=$MEDAKA_ROUNDS')
    print('\n'.join(seqlines))
" > "$final_polished"

# Check final output was generated and create link
if [ ! -f "$final_polished" ]; then
    echo "❌ ERROR: Final polished assembly was NOT created: $final_polished"
    exit 1
fi

echo "✅ Polished assembly created: $final_polished"

# Create final output link
ln -sf "$final_polished" "$final_output"
echo "✅ Medaka polishing complete: ${final_output} -> ${final_polished}"
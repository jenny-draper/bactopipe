#!/bin/bash
"""
Medaka Polishing - BACTOPIPE Medaka Polishing Script
==========================================================================

Runs Medaka consensus polishing on bacterial assemblies.
Automatically detects the appropriate Medaka model from QC run output where available.

Author: Jenny Draper
Date: October 2025

Usage:
    run_medaka.sh <SAMPLE_ID> <RUNID> <ASSEMBLY> <READS>
    run_medaka.sh --version
"""

set -euo pipefail

VERSION="1.0"

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

# Set up directories
medaka_dir="/data/runs/ont/analysis/${RUNID}/assembly/medaka/${SAMPLE_ID}"
assembly_dir="/data/runs/ont/analysis/${RUNID}/assembly"
final_output="${assembly_dir}/${SAMPLE_ID}.fa"
mkdir -p "$medaka_dir"

# Determine Medaka model for this run
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

# Create final polished file with preserved headers
final_polished="${medaka_dir}/${SAMPLE_ID}.polished.fa"

# Extract original headers and preserve them with medaka annotation
awk -v medaka_rounds="$MEDAKA_ROUNDS" '
BEGIN { 
    # Read original headers from assembly file
    while ((getline line < "'$assembly'") > 0) {
        if (line ~ /^>/) {
            orig_headers[++header_count] = substr(line, 2) " medaka_polish=" medaka_rounds
        }
    }
    close("'$assembly'")
    contig_num = 0
}
/^>/ { 
    contig_num++
    if (contig_num in orig_headers) {
        print ">" orig_headers[contig_num]
    } else {
        print $0 " medaka_polish=" medaka_rounds
    }
    next
}
{ print }
' "$current_fa" > "$final_polished"

# Check final output and create link
if [ ! -f "$final_polished" ]; then
    echo "❌ ERROR: Final polished assembly not created: $final_polished"
    exit 1
fi
ln -sf "$final_polished" "$final_output"
echo "✅ Medaka polishing complete: ${SAMPLE_ID}.fa"
echo "$final_output"  # Output final path for pipeline to use
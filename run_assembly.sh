#!/bin/bash
"""
Assembly - BACTOPIPE Bacterial Genome Assembly Script
=============================================================================

Performs bacterial genome assembly using Autocycler, with a fallback to 
Dragonflye if Autocycler fails. 

Note it this does NOT perform medaka polishing; that has been moved to a 
separate step, to enable higher parallelism in the assembly pipeline.

Author: Jenny Draper
Date: October 2025

Usage:
    run_assembly.sh <SAMPLE_ID> <RUNID>
    run_assembly.sh --version
"""

set -euo pipefail

VERSION="1.0"

# Handle version argument
[[ "${1:-}" == "--version" ]] && echo "$(basename "$0") ${VERSION}" && exit 0

# Defaults
AUTOCYCLER_DEPTH=400
DRAGONFLYE_DEPTH=100

# Parse arguments
SAMPLE_ID=$1
RUNID=$2

# Validate arguments
if [ -z "$SAMPLE_ID" ] || [ -z "$RUNID" ]; then
    echo "❌ ERROR: Missing arguments. Usage: $0 <SAMPLE_ID> <RUNID>"
    exit 1
fi

# Load modules
module purge
module load trycycler
module load dragonflye

# Set up paths
QCDIR="/data/runs/ont/qc/${RUNID}"
ASSEMBLYDIR="/data/runs/ont/analysis/${RUNID}/assembly"
reads="${QCDIR}/fastq_pass_concatenated/${SAMPLE_ID}.fastq.gz"

# Get genome size from samples file
size=$(awk -F'\t' -v id="$SAMPLE_ID" '$2==id {print $6}' "/data/runs/ont/analysis/${RUNID}/samples.tsv")

echo "Starting assembly for SAMPLE: $SAMPLE_ID (RUN: $RUNID)"

# Check required files
if [ ! -f "$reads" ]; then
    echo "❌ ERROR: Reads not found: $reads"
    exit 1
fi

# Function to check for assembly output files
find_best_assembly() {
    local dir="$1"
    local prefix="$2"
    for suffix in ".reoriented.fa" ".fa"; do
        local fa="${dir}/${prefix}${suffix}"
        if [ -f "$fa" ]; then
            echo "$fa"
            return 0
        fi
    done
    return 1
}

# Build assembly options
base_opts="--trim --keepfiles --nanohq --seed 42 --racon 2 --medaka 0"
size_opt=""
if [ -n "$size" ] && [ "$size" != "NA" ]; then
    size_opt="--gsize $size"
fi

# Try Autocycler first
autocycler_dir="${ASSEMBLYDIR}/autocycler/${SAMPLE_ID}"
mkdir -p "$autocycler_dir"
autocycler_prefix="${SAMPLE_ID}-dragonflye_auto"

echo "▶️ Attempting Autocycler assembly"
if dragonflye_auto \
    --assembler autocycler \
    --reads "$reads" \
    --outdir "$autocycler_dir" \
    --prefix "$autocycler_prefix" \
    --depth "$AUTOCYCLER_DEPTH" \
    --force \
    $base_opts $size_opt; then
    
    if best_assembly=$(find_best_assembly "$autocycler_dir" "$autocycler_prefix"); then
        echo "✅ Autocycler succeeded: $(basename "$best_assembly")"
    fi
fi

# Fallback to Dragonflye if autocycler failed
if [ -z "${best_assembly:-}" ]; then
    echo "⚠️ Autocycler failed, trying Dragonflye"
    dragonflye_dir="${ASSEMBLYDIR}/dragonflye/${SAMPLE_ID}"
    mkdir -p "$dragonflye_dir"
    dragonflye_prefix="${SAMPLE_ID}-dragonflye"

    # Use --meta flag if no genome size
    dragonflye_size_opt="$size_opt"
    if [ -z "$size_opt" ]; then
        dragonflye_size_opt='--opts "--meta"'
    fi

    if dragonflye \
        --reads "$reads" \
        --outdir "$dragonflye_dir" \
        --prefix "$dragonflye_prefix" \
        --depth "$DRAGONFLYE_DEPTH" \
        --force \
        $base_opts $dragonflye_size_opt; then
        
        if best_assembly=$(find_best_assembly "$dragonflye_dir" "$dragonflye_prefix"); then
            echo "✅ Dragonflye succeeded: $(basename "$best_assembly")"
        fi
    fi
fi

# Create final symbolic link to the best assembly
final_link="${ASSEMBLYDIR}/${SAMPLE_ID}.unpolished.fa"

if [ -n "${best_assembly:-}" ] && [ -f "$best_assembly" ]; then
    ln -sf "$best_assembly" "$final_link"
    echo "✅ Created final assembly link: ${SAMPLE_ID}.unpolished.fa -> $(basename "$best_assembly")"
    echo "Final assembly: $final_link"
else
    echo "❌ No suitable assembly found for $SAMPLE_ID"
    exit 1
fi
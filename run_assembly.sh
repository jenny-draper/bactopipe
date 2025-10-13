#!/bin/bash

# Assembly - BACTOPIPE Bacterial Genome Assembly Script
# ============================================================================
#
# Performs bacterial genome assembly using Autocycler, with a fallback to
# Dragonflye if Autocycler fails.
#
# Note it this does NOT perform medaka polishing; that has been moved to a
# separate step, to enable higher parallelism in the assembly pipeline.
#
# Author: Jenny Draper
#       Date: October 2025
#   
# Usage:
#    run_assembly.sh <SAMPLE_ID> <RUNID>
#    run_assembly.sh --version
#

set -euo pipefail

VERSION="1.1"

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

# Check required files
if [ ! -f "$reads" ]; then echo "❌ ERROR: Reads not found: $reads"; exit 1; fi

echo "Starting assembly for SAMPLE: $SAMPLE_ID (RUN: $RUNID)"

# Get genome size from samples file
gsize=$(awk -F'\t' -v id="$SAMPLE_ID" '$2==id {print $6}' "/data/runs/ont/analysis/${RUNID}/samples.tsv")

# Function to find best assembly and report success
find_assembly() {
    local dir="$1"
    local prefix="$2"
    local assembler="$3"
    
    best_assembly="${dir}/${prefix}.reoriented.fa"
    [ ! -f "$best_assembly" ] && best_assembly="${dir}/${prefix}.fa"
    [ -f "$best_assembly" ] && echo "✅ ${assembler} succeeded: $(basename "$best_assembly")"
}

# Build assembly options
base_opts="--trim --keepfiles --nanohq --seed 42 --racon 2 --medaka 0"

# Try Autocycler first ----------------------------------------------------------------
autocycler_dir="${ASSEMBLYDIR}/autocycler/${SAMPLE_ID}"
mkdir -p "$autocycler_dir"
autocycler_prefix="${SAMPLE_ID}.autocycler"

echo "▶️ Attempting Autocycler assembly"
autocycler_cmd="dragonflye_auto \
                     --assembler autocycler \
                     --reads $reads \
                     --outdir $autocycler_dir \
                     --prefix $autocycler_prefix \
                     --depth $AUTOCYCLER_DEPTH \
                     --force \
                     $base_opts"
if [ -n "$gsize" ] && [ "$gsize" != "NA" ]; then autocycler_cmd="$autocycler_cmd --gsize $gsize"; fi

echo "Running: $autocycler_cmd"
if eval "$autocycler_cmd"; then
    find_assembly "$autocycler_dir" "$autocycler_prefix" "Autocycler"
fi

# Fallback to Dragonflye if autocycler failed ---------------------------------------
if [ -z "${best_assembly:-}" ]; then
    echo "⚠️ Autocycler failed, trying Dragonflye"
    dragonflye_dir="${ASSEMBLYDIR}/dragonflye/${SAMPLE_ID}"
    mkdir -p "$dragonflye_dir"
    dragonflye_prefix="${SAMPLE_ID}.dragonflye"

    # set up dragonflye cmd & -gsize/-meta mode 
    dragonflye_cmd="dragonflye \
                        --reads $reads \
                        --outdir $dragonflye_dir \
                        --prefix $dragonflye_prefix\
                         --depth $DRAGONFLYE_DEPTH \
                        --force \
                         $base_opts"
    if [ -n "$gsize" ] && [ "$gsize" != "NA" ]; then
        dragonflye_cmd="$dragonflye_cmd --gsize $gsize"
    else
        dragonflye_cmd="$dragonflye_cmd --opts \"--meta\""
    fi

    echo "Running: $dragonflye_cmd"
    if eval "$dragonflye_cmd"; then
        find_assembly "$dragonflye_dir" "$dragonflye_prefix" "Dragonflye"
    fi
fi

# Create final symbolic link to the best assembly
mkdir -p "${ASSEMBLYDIR}/unpolished_best"

if [ -n "${best_assembly:-}" ] && [ -f "$best_assembly" ]; then
    # Determine assembly method from the actual assembly filename
    assembly_method=$(basename "$best_assembly" .fa)
    assembly_method="${assembly_method#${SAMPLE_ID}.}"
    
    # Replace assembly method strings and join with hyphens
    assembly_method="${assembly_method//dragonflye_auto/autocycler}"
    assembly_method="${assembly_method//reoriented/reori}"
    assembly_method="${assembly_method//dragoneflye/dragonflye}"
    assembly_method="${assembly_method//./-}"  # Replace dots with hyphens
    
    # Create descriptive link name only
    final_link="${ASSEMBLYDIR}/unpolished_best/${SAMPLE_ID}.${assembly_method}-unpolished.fa"
    
    ln -sf "$best_assembly" "$final_link"

    echo "✅ Created best assembly link: ${final_link} -> ${best_assembly}"
else
    echo "❌ No suitable assembly found for $SAMPLE_ID"
    exit 1
fi
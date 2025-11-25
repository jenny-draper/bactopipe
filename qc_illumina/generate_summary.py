#!/usr/bin/env python3
"""
Parse MultiQC JSON data and generate QC summary TSV
"""

import json
import argparse
from pathlib import Path
import sys
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse MultiQC data and generate a summary TSV.")
    parser.add_argument("-p", "--path", required=True, help="Path to the QC directory containing MultiQC data.")
    parser.add_argument("--version", action="version", version="1.0")
    return parser.parse_args()

def extract_sample_metrics(sample_id, fastqc_data, trimmomatic_data, kraken_data, fastqc_raw_data):
    """Extract and format metrics for a single sample."""
    
    # Remove suffixes to get base sample ID
    base_sample_id = sample_id.replace("_processed_R1", "").replace("_processed_R2", "")
    
    # FastQC metrics
    metrics = fastqc_data.get(sample_id, {})
    total_sequences = int(metrics.get("total_sequences", 0)) * 2
    percent_gc = round(float(metrics.get("percent_gc", 0)), 1)
    avg_sequence_length = round(float(metrics.get("avg_sequence_length", 0)), 1)
    
    # Trimmomatic survival percentage
    percent_surviving = 0.0
    if base_sample_id in trimmomatic_data:
        percent_surviving = round(float(trimmomatic_data[base_sample_id].get("surviving_pct", 0)), 2)
    
    # Kraken classification results - MultiQC keys have .kraken suffix
    kraken_hits = []
    for possible_id in [f"{base_sample_id}.kraken", base_sample_id, f"{base_sample_id}_processed"]:
        if possible_id in kraken_data:
            kraken_hits = kraken_data[possible_id]
            break
    
    # Unclassified percentage
    unclassified_percent = 0.0
    for hit in kraken_hits:
        if hit.get("rank_code") == "U":
            unclassified_percent = round(hit["percent"], 1)
            break
    
    # Top genera
    genus_hits = [hit for hit in kraken_hits if hit.get("rank_code") == "G"]
    top_genus = genus_hits[0]["classif"] if genus_hits else "N/A"
    top_genus_percent = round(genus_hits[0]["percent"], 2) if genus_hits else 0.00
    second_genus = genus_hits[1]["classif"] if len(genus_hits) > 1 else "N/A"
    second_genus_percent = round(genus_hits[1]["percent"], 2) if len(genus_hits) > 1 else 0.00
    
    # Top species
    species_hits = [hit for hit in kraken_hits if hit.get("rank_code") == "S"]
    top_species = species_hits[0]["classif"] if species_hits else "N/A"
    top_species_percent = round(species_hits[0]["percent"], 2) if species_hits else 0.00
    second_species = species_hits[1]["classif"] if len(species_hits) > 1 else "N/A"
    second_species_percent = round(species_hits[1]["percent"], 2) if len(species_hits) > 1 else 0.00
    
    # FastQC failures (excluding per_base_sequence_content)
    fastqc_fail_stats = set()
    for read_type in ["_processed_R1", "_processed_R2"]:
        fastqc_sample_key = f"{base_sample_id}{read_type}"
        if fastqc_sample_key in fastqc_raw_data:
            for category, status in fastqc_raw_data[fastqc_sample_key].items():
                if status == "fail" and category != "per_base_sequence_content":
                    fastqc_fail_stats.add(category)
    
    return {
        "Sample_ID": base_sample_id,
        "Total_Trimmed_Reads": total_sequences,
        "GC_PCT": percent_gc,
        "Avg_Sequence_Length": avg_sequence_length,
        "Passing_Trim_PCT": percent_surviving,
        "Unclassified_PCT": unclassified_percent,
        "Top_Genus": top_genus,
        "Top_Genus_PCT": top_genus_percent,
        "Second_Genus": second_genus,
        "Second_Genus_PCT": second_genus_percent,
        "Top_Species": top_species,
        "Top_Species_PCT": top_species_percent,
        "Second_Species": second_species,
        "Second_Species_PCT": second_species_percent,
        "FASTQC_fails": "; ".join(sorted(fastqc_fail_stats)) if fastqc_fail_stats else ""
    }

def main():
    args = parse_arguments()
    
    qcdir = Path(args.path)
    input_json_path = qcdir / "multiqc/multiqc_data/multiqc_data.json"
    species_file = qcdir / "kraken2/species_identified.tsv"
    final_output = qcdir / "qc_summary.tsv"
    samples_file = qcdir / "samples.tsv"
    
    # Check input files exist
    for infile in [input_json_path, species_file, samples_file]:
        if not infile.exists():
            sys.exit(f"Error: Required input file not found: {infile}")
    
    # Read and parse JSON
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Failed to parse JSON file: {e}")
    
    # Extract data sections
    try:
        fastqc_data = data["report_general_stats_data"][2]
        kraken_data = data["report_saved_raw_data"]["multiqc_kraken"]
        trimmomatic_data = data["report_saved_raw_data"]["multiqc_trimmomatic"]
        fastqc_raw_data = data["report_saved_raw_data"]["multiqc_fastqc"]
    except (KeyError, IndexError) as e:
        sys.exit(f"Error: Required data section not found in MultiQC JSON: {e}")
    
    # Read sample IDs from samples.tsv using pandas
    samples_df = pd.read_csv(samples_file, sep='\t')
    sample_ids = samples_df['SAMPLE_ID'].dropna().tolist()
    
    # Load species data
    species_df = pd.read_csv(species_file, sep='\t')
    species_dict = species_df.set_index('Sample_ID').to_dict('index')
    
    # Process all samples in one loop
    results = []
    for sample_id in sample_ids:
        # Extract QC metrics
        fastqc_key = f"{sample_id}_processed_R1"
        qc_metrics = extract_sample_metrics(
            fastqc_key, fastqc_data, trimmomatic_data, kraken_data, fastqc_raw_data
        )
        
        # Merge with species data
        if sample_id in species_dict:
            qc_metrics.update(species_dict[sample_id])
        else:
            # Add empty species columns if not found
            qc_metrics['TaxID'] = 'NA'
            qc_metrics['Expected_Genome_Size'] = 'NA'
        
        results.append(qc_metrics)
    
    # Write final output
    df = pd.DataFrame(results)
    df.to_csv(final_output, sep='\t', index=False)
    print(f"Final QC summary with species info: {final_output}")

if __name__ == '__main__':
    main()

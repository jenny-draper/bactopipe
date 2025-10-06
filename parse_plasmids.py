#!/usr/bin/python3
"""
Parse Plasmids - BACTOPIPE Plasmid Summary Tool
=================================================================

Maps PlasmidFinder and abricate AMR hits to specific contigs for 
comprehensive plasmid-associated resistance gene analysis.

Author: Jenny Draper
Date: June 2025

Usage:
    parse_plasmids.py -r RUN_ID
"""

#=======================================================================================================

import argparse
import pandas as pd
import os
import subprocess

VERSION = "1.0"

# Set pandas options to display all columns and prevent wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Script to map PlasmidFinder and AbriTAMR hits to specific contigs.")
parser.add_argument("-r", "--runid", help="Name of the run to process. All other path aspects are automated.")
parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
args = parser.parse_args()


runid = args.runid
rundir = f"/data/runs/ont/analysis/{runid}"

print(f"\n\nIdentifying plasmid contigs in run: {runid}")

# Read sample sheets (includes species ID from QC kraken)
samplesheet_path = f"{rundir}/samples.tsv"
print(f"Reading sample sheet: {samplesheet_path}")
samples = pd.read_csv(samplesheet_path, sep="\t")
samples["RUNID"] = runid
samples = samples[["RUNID", "SAMPLE_ID"]]


def extract_contig_info(contig, header_line, rundir, sample_id):
    """Extract contig length and circularity from header line"""
    try:
        if "autocycler" in header_line:
            # Autocycler format - extract from header directly
            header_parts = header_line.split()
            length_parts = [p for p in header_parts if p.startswith("len=")]
            rotated_parts = [p for p in header_parts if p.startswith("rotated=")]
            
            if not length_parts:
                return "NA", "NA"
            
            contig_length = int(length_parts[0].split("=")[1])
            
            # Determine circularity from rotated status
            if rotated_parts:
                is_rotated = rotated_parts[0].split("=")[1].lower() == "true"
                circularity = "Y" if is_rotated else "N"
            else:
                circularity = "N"
            
            return contig_length, circularity
            
        else:
            # Dragonflye format - search for specific patterns
            header_parts = header_line.split()
            
            # Find length
            length_parts = [p for p in header_parts if p.startswith("len=")]
            if not length_parts:
                return "NA", "NA"
            contig_length = int(length_parts[0].split("=")[1])
            
            # Find circularity - look for circular= pattern
            circular_parts = [p for p in header_parts if p.startswith("circular=")]
            if circular_parts:
                circularity = circular_parts[0].split("=")[1]
            else:
                circularity = "N"  # Default to non-circular if not found

            return contig_length, circularity
                
    except (IndexError, ValueError, Exception):
        return "NA", "NA"


# Parse amrfinder & plasmidfinder results per isolate & join into one df
print("Parsing result files ...")
retdf = pd.DataFrame(columns=["SAMPLE_ID", "Contig", "Plasmidfinder", "AMRfinder", "Contig_length"])

for id in samples["SAMPLE_ID"]:
    # Updated paths to match current pipeline structure
    assemblyf = f"{rundir}/assembly/{id}.fa"
    amrfinderf = f"{rundir}/abritamr/{id}/amrfinder.out"
    plasmidfinderf = f"{rundir}/plasmidfinder/{id}/results_tab.tsv"

    # Check if the required files exist
    missing_files = []
    for file_path, file_type in [(assemblyf, "assembly"), (amrfinderf, "amrfinder"), (plasmidfinderf, "plasmidfinder")]:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {file_path}")
        
    if missing_files: 
        print(f"{id}: WARNING: missing files: {', '.join(missing_files)}")
        retdf = pd.concat([retdf, pd.DataFrame({"SAMPLE_ID": [id], "Contig": ["NA"], "Plasmidfinder": ["NA"], "AMRfinder": ["NA"], "Contig_length": ["NA"], "Contig_circular": ["NA"]})], ignore_index=True)
        continue

    print(f"{id}: assembly: {os.path.realpath(assemblyf)}")
    amrfinder = pd.read_csv(amrfinderf, sep="\t")
    plasmidfinder = pd.read_csv(plasmidfinderf, sep="\t")

    # Prep plasmidfinder df
    plasmidfinder.rename(columns={"Contig": "Contig_full"}, inplace=True)  # has full contig header details
    plasmidfinder["Contig"] = plasmidfinder["Contig_full"].str.split().str[0]  # extract contig id only
    plasmidfinder["Plasmid_Accession"] = plasmidfinder["Plasmid"] + "_" + plasmidfinder["Accession number"]
    plasmidfinder = plasmidfinder[["Contig", "Plasmid", "Plasmid_Accession"]]

    # Prep amrfinder df
    amrfinder.rename(columns={"Gene symbol": "Gene", "Contig id": "Contig"}, inplace=True)
    amrfinder = amrfinder[["Gene", "Contig"]]

    # Merge the DataFrames on the Contig column - use outer join to get all contigs
    merged_df = pd.merge(amrfinder, plasmidfinder, on="Contig", how="outer")
    
    # Check if we have any results at all
    if merged_df.empty:
        print(f"{id}: WARNING: No AMR or plasmid hits found for sample {id}, adding 'none' entry")
        retdf = pd.concat([retdf, pd.DataFrame({"SAMPLE_ID": [id], "Contig": ["none"], "Plasmidfinder": ["none"], "AMRfinder": ["none"], "Contig_length": ["NA"], "Contig_circular": ["NA"]})], ignore_index=True)
        continue

    # Aggregate the data
    aggregated_df = merged_df.groupby("Contig").agg({
        "Plasmid": lambda x: ",".join(sorted(x.dropna().unique())),
        "Gene": lambda x: ",".join(sorted(x.dropna().unique()))
    }).reset_index()

    # Extract contig info from assemblyf for each contig 
    # (can't use PF output as not every AMR contig is a plasmid contig)
    contig_lengths = []
    contig_circular = []
    for contig in aggregated_df["Contig"]:
        # Run grep to extract the fasta header line with the contig id
        result = subprocess.run(['grep', f'>{contig}', assemblyf], capture_output=True, text=True)
        header_line = result.stdout.strip()

        if header_line:
            length, circularity = extract_contig_info(contig, header_line, rundir, id)
        else:
            length, circularity = "NA", "NA"
        
        contig_lengths.append(length)
        contig_circular.append(circularity)
        
    aggregated_df["Contig_length"] = contig_lengths
    aggregated_df["Contig_circular"] = contig_circular

    # Rename the aggregated columns
    aggregated_df.rename(columns={"Plasmid": "Plasmidfinder", "Gene": "AMRfinder"}, inplace=True)

    # Add SAMPLE_ID column
    aggregated_df.insert(0, 'SAMPLE_ID', id)

    # Append to retdf
    retdf = pd.concat([retdf, aggregated_df], ignore_index=True)

# Ensure Contig_length is either an integer or "NA"
retdf["Contig_length"] = retdf["Contig_length"].apply(lambda x: int(x) if x != "NA" else "NA")

print("\nFinal combined DataFrame:")
print(retdf)

# Save the final DataFrame to a TSV file
outfile = f"{rundir}/{runid}.plasmid_contigs.tsv"
retdf.to_csv(outfile, sep="\t", index=False)
print(f"Done. Results saved to: {outfile}")

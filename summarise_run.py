#!/usr/bin/python3
"""
Summarise Run - BACTOPIPE Results Aggregator
============================================================================

Merges ONT run QC and analysis outputs into a single summary table.
Combines assembly statistics, MLST, CheckM, PlasmidFinder, and AMR results.

Author: J. Draper
Date: August 2024

Usage:
    summarise_run.py -r RUN_ID
"""

#=======================================================================================================

import argparse
import pandas as pd
import re
import os
import yaml

VERSION = "1.1"

parser = argparse.ArgumentParser(description="Script to combine the key analysis & QC output for a run into one table.")
parser.add_argument("-r", "--runid", help="Name of the run to process. All other path aspects are automated.")
parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
args = parser.parse_args()

runid = args.runid

qcdir = "/data/qc/" + runid
rundir = "/data/runs/ont/analysis/" + runid

print("Generating final summary table for run "+runid)

# Sample sheets (includes species ID from QC kraken)
print("reading sample sheet")
samples = pd.read_csv(rundir + "/samples.tsv", sep="\t")
samples["RUNID"] = runid
samples = samples[["RUNID", "SAMPLE_ID", "BARCODE", "EXPECTED_SPECIES", "IDENTIFED_SPECIES", "IDENTIFIED_GENOME_SIZE"]]
samples.rename(columns=lambda x: x.lower() if x in ["EXPECTED_SPECIES", "IDENTIFED_SPECIES", "IDENTIFIED_GENOME_SIZE"] else x, inplace=True)

# Remove any duplicates in the samples file
if samples.duplicated(subset=['SAMPLE_ID']).any():
    print(f"WARNING: Found duplicate SAMPLE_IDs in samples.tsv, keeping first occurrence")
    samples = samples.drop_duplicates(subset=['SAMPLE_ID'], keep='first')

# Get metrics values (N50, read count, total bases, etc) 
print("collecting QC data")
qc = pd.read_csv(qcdir + "/" + runid + ".qc_result_table.csv")
qc = qc[["SAMPLE_ID", "number_of_reads", "number_of_bases", "median_read_length", "n50", "median_qual", "Unclassified.Percentage"]]
merged = pd.merge(samples, qc, on="SAMPLE_ID", how="left")

# Handle NaN values in IDENTIFIED_GENOME_SIZE
merged['identified_genome_size'] = pd.to_numeric(merged['identified_genome_size'], errors='coerce')
merged['identified_genome_size'] = merged['identified_genome_size'].fillna(0).astype(int)

# Calculate coverage estimate based on genome size estimate for the actual species
def calc_coverage(row):
    if row["identified_genome_size"] > 0:
        return int(row["number_of_bases"] / row["identified_genome_size"])
    else:
        return 0
merged["coverage"] = merged.apply(calc_coverage, axis=1)


def get_autocycler_circularity(rundir, sample_id, contig_id):
    """Get circularity info from autocycler YAML file"""
    try:
        yaml_file = f"{rundir}/assembly/autocycler/{sample_id}/assembly_graph_summary.yaml"
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Find the contig in the YAML data
            for contig_info in data.get('Contig_info', []):
                if contig_info.get('Name') == contig_id:
                    topology = contig_info.get('Topology', 'linear')
                    return "Y" if topology == "circular" else "N"
        
        return None  # Could not determine from YAML
    except Exception:
        return None


def parse_contig_header(header, rundir=None, sample_id=None):
    """Parse contig header to extract ID, length, coverage, and circularity"""
    try:
        # Extract contig ID (everything before first space)
        cid_match = re.search(r">(\S+)", header)
        if not cid_match:
            return None, None, None, None
        cid = cid_match[1]
        
        # Extract length (present in both formats)
        length_match = re.search(r"len=([0-9]+)", header)
        length = int(length_match[1]) if length_match else None
        
        # Extract coverage (only in dragonflye format)
        cov_match = re.search(r"cov=([0-9.]+)", header)
        cov = int(float(cov_match[1])) if cov_match else 0
        
        # Extract circularity
        # Check for explicit circular=Y (dragonflye format)
        circular_match = re.search(r"circular=([YN])", header)
        if circular_match:
            circularity = circular_match[1]
        else:
            # For autocycler format, get circularity from YAML file
            if rundir and sample_id:
                circularity = get_autocycler_circularity(rundir, sample_id, cid)
                if circularity is not None:
                    return cid, length, cov, circularity
            
            # Use NA if YAML lookup fails
            circularity = "NA"
        
        return cid, length, cov, circularity
            
    except (IndexError, ValueError, AttributeError):
        return cid if 'cid' in locals() else None, None, None, None

# Assembly stats
print("collecting assembly info")
merged["contigs"] = ""
merged["num_contigs"] = int(0)
merged["longest_contig"] = int(0)
merged["assembly_method"] = ""

# Extract contig info from final assembly files
for id in merged["SAMPLE_ID"]:
    contig_tuples = []  # (length, contig_str)
    longest = 0
    assemblyf = f"{rundir}/assembly/{id}.fa"

    if not os.path.isfile(assemblyf):
        print(f"WARNING: {assemblyf} not found, setting contig info to NA for sample {id}")
        continue

    # Extract assembly method from the symlink target
    try:
        if os.path.islink(assemblyf):
            # Follow the link and extract method from filename
            target = os.readlink(assemblyf)
            # Target will be like: medaka/{id}/{id}.autocycler-reori-polished.fa
            target_basename = os.path.basename(target)
            # Remove sample_id prefix and .fa suffix
            method_part = target_basename.replace(f"{id}.", "").replace(".fa", "")
            # Remove "-polished" to get just the assembly method
            assembly_method = method_part.replace("-polished", "")
        else:
            assembly_method = "unknown"
    except Exception as e:
        print(f"WARNING: Could not determine assembly method for {id}: {e}")
        assembly_method = "unknown"

    with open(assemblyf, 'r') as f:
        all_contigs = [line for line in f if ">" in line]

    for header in all_contigs:
        cid, length, cov, circularity = parse_contig_header(header, rundir, id)
        # Skip if we couldn't parse the header properly
        if cid is None or length is None:
            print(f"WARNING: Could not parse header for sample {id}: {header.strip()}")
            continue
        if longest < length:
            longest = length
        contig = f"{cid}.{length}bp.{cov}x"
        contig = contig + ".c" if circularity == "Y" else contig + ".l"
        contig_tuples.append((length, contig))

    # Sort by length descending
    contig_tuples.sort(reverse=True)
    contigs_sorted = [c[1] for c in contig_tuples]
    contigs_display = contigs_sorted[:10] + (["[etc]"] if len(contigs_sorted) > 10 else [])

    sample_idx = merged.loc[merged['SAMPLE_ID'] == id].index[0]
    merged.at[sample_idx, "contigs"] = ", ".join(contigs_display)
    merged.at[sample_idx, "num_contigs"] = len(contigs_sorted)
    merged.at[sample_idx, "longest_contig"] = longest
    merged.at[sample_idx, "assembly_method"] = assembly_method

# MLST results
print("collecting mlst info")
mlst = pd.read_fwf(rundir+"/mlst/mlst.csv", header=None)
mlst = mlst[0].str.split(',', expand=True)[[0,1,2]]
mlst.columns=["SAMPLE_ID", "MLST_scheme", "ST"]

# Strip .fa and .unpolished.fa suffixes from sample IDs
mlst["SAMPLE_ID"] = mlst["SAMPLE_ID"].str.replace(r"\.(unpolished\.)?fa$", "", regex=True)

merged = pd.merge(merged, mlst, on="SAMPLE_ID", how="left")

# CheckM results
print("collecting checkm stats")
checkm = pd.read_csv( rundir+"/checkm/checkm_results.tsv", sep="\t" )
checkm = checkm.rename(columns={"Bin Id": "SAMPLE_ID", 
                                "Completeness": "checkm_completeness", 
                                "Contamination": "checkm_contamination", 
                                "Strain heterogeneity": "checkm_heterogeneity",
                                "Genome size (bp)": "assembly_size_bp"})
merged = pd.merge(merged, checkm[["SAMPLE_ID", "assembly_size_bp", "checkm_completeness", "checkm_contamination","checkm_heterogeneity"]], on="SAMPLE_ID", how="left")

# PlasmidFinder results
print("collecting plasmid info")
plasmids = pd.read_csv(rundir+"/plasmidfinder/plasmidfinder_results.tsv", sep="\t", on_bad_lines='warn')
plasmids = plasmids.rename(columns={"Sample": "SAMPLE_ID"})
plasmids = plasmids.fillna("NA").astype(str)
plasmids = plasmids.groupby('SAMPLE_ID')['Plasmid'].apply(lambda x: ', '.join(x)).reset_index()
plasmids.columns = ['SAMPLE_ID', 'Plasmids']
merged = pd.merge(merged, plasmids, on="SAMPLE_ID", how="left")

# AMRFinder results
print("collecting amr info")
amr = pd.read_csv( rundir+"/abritamr/summary_matches.txt", sep="\t" )
amr = amr.rename(columns={"Isolate": "SAMPLE_ID"})
merged = pd.merge(merged, amr, on="SAMPLE_ID", how="left")

# Remove any duplicate rows that may have been created during merging
merged = merged.drop_duplicates()

# Drop rows that are all NaN (except for the key columns)
key_columns = ["RUNID", "SAMPLE_ID", "BARCODE"]
non_key_columns = [col for col in merged.columns if col not in key_columns]
merged = merged.dropna(subset=non_key_columns, how='all')

# Save results
outfile = f"{rundir}/{runid}.summary.tsv"
merged.to_csv(outfile, sep="\t", index=False)

print("Done. Wrote summary file: "+outfile)
print(merged)


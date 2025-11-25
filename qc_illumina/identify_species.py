#!/usr/bin/env python3
"""
Parse single Kraken2 report and identify species with NCBI genome size lookup
"""

import argparse
import pandas as pd
import urllib.request
from pathlib import Path
import sys
import re


def parse_kraken_report(report_file, output_file):
    """Parse Kraken2 report and save top hits."""
    # Read and filter relevant lines
    data = []
    with open(report_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) >= 6 and cols[3] in ['G', 'S', 'U']:
                data.append(cols)
    
    if not data:
        output_file.write_text("Percentage,Count_Clade,Count_Taxon,TaxRank,TaxID,Organism\n")
        return None, None
    
    # Process data
    df = pd.DataFrame(data, columns=['Percentage', 'Count_Clade', 'Count_Taxon', 
                                     'TaxRank', 'TaxID', 'Name'])
    df['Organism'] = df['Name'].str.strip()
    
    # Get top results
    top_results = pd.concat([
        df[df['TaxRank'] == 'G'].head(3),
        df[df['TaxRank'] == 'S'].head(3),
        df[df['TaxRank'] == 'U'].head(1)
    ])
    
    # Clean up and save
    top_results['TaxRank'] = top_results['TaxRank'].map({
        'G': 'Genus', 'S': 'Species', 'U': 'Unclassified'
    })
    top_results[['Percentage', 'Count_Clade', 'Count_Taxon', 'TaxRank', 'TaxID', 'Organism']].to_csv(
        output_file, index=False
    )
    
    # Return top species if found
    species = df[df['TaxRank'] == 'S']
    if not species.empty:
        top = species.iloc[0]
        return top['Organism'].strip(), top['TaxID']
    
    return None, None


def lookup_genome_size(species_name, taxid, db_file):
    """Lookup genome size from cache or fetch from NCBI."""
    # Check cache first
    if db_file.exists():
        for line in db_file.read_text().splitlines()[1:]:  # Skip header
            parts = line.split('\t')
            if len(parts) >= 3 and parts[0] == species_name:
                return parts[2]
    
    # Fetch from NCBI
    try:
        url = f"https://api.ncbi.nlm.nih.gov/genome/v0/expected_genome_size?species_taxid={taxid}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read().decode('utf-8')
            match = re.search(r'expected_ungapped_length>([^<]+)', data)
            size = match.group(1) if match else "NA"
            
            # Cache valid results
            if size != "NA":
                with open(db_file, 'a') as f:
                    f.write(f"{species_name}\t{taxid}\t{size}\n")
            return size
            
    except Exception as e:
        print(f"Warning: Could not fetch genome size for {species_name}: {e}", file=sys.stderr)
        return "NA"


def main():
    parser = argparse.ArgumentParser(
        description='Parse single Kraken2 report and identify species with genome size lookup'
    )
    parser.add_argument('--sample-id', required=True, help='Sample ID')
    parser.add_argument('--report', required=True, help='Kraken2 report file')
    parser.add_argument('--output', required=True, help='Output TSV file for this sample')
    parser.add_argument('--size-db', required=True, help='Genome size database file')
    parser.add_argument('--rename-db', help='Species rename mapping file (optional)')
    parser.add_argument('--version', action='version', version='2.8')
    
    args = parser.parse_args()
    
    report_file = Path(args.report)
    
    # Ensure database exists
    db_file = Path(args.size_db)
    if not db_file.exists():
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_text("species\ttaxid\tgenome_size\n")
    
    # Load species renames if provided
    rename_dict = {}
    if args.rename_db:
        try:
            df = pd.read_csv(args.rename_db, sep='\t')
            rename_dict = dict(zip(df['old_name'], df['new_name']))
        except Exception as e:
            print(f"Warning: Could not load rename file {args.rename_db}: {e}", file=sys.stderr)
    
    # Default result
    result = {
        'Sample_ID': args.sample_id,
        'Top_Species': 'Unknown',
        'TaxID': 'NA',
        'Expected_Genome_Size': 'NA'
    }
    
    if not report_file.exists():
        print(f"Warning: Kraken2 report not found: {report_file}", file=sys.stderr)
    else:
        # Create tophits output path (same dir as output)
        output_path = Path(args.output)
        tophits_file = output_path.parent / f"{args.sample_id}.kraken.tophits.csv"
        
        # Parse report
        species_name, taxid = parse_kraken_report(report_file, tophits_file)
        if species_name:
            # Apply rename if applicable
            species_name = rename_dict.get(species_name, species_name)
            
            genome_size = lookup_genome_size(species_name, taxid, db_file)
            result = {
                'Sample_ID': args.sample_id,
                'Top_Species': species_name,
                'TaxID': taxid,
                'Expected_Genome_Size': genome_size
            }
    
    # Write single-sample output
    df = pd.DataFrame([result])
    df.to_csv(args.output, sep='\t', index=False)
    
    print(f"Processed {args.sample_id}: {result['Top_Species']}")


if __name__ == '__main__':
    main()

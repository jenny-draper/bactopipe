# BactoPipe:
## ONT Bacterial Genome Analysis Pipeline

**A flexible, YAML-configured workflow-like pipeline for bacterial genome analysis from Oxford Nanopore sequencing data.**

BactoPipe automates the complete analysis of bacterial genomes from raw ONT sequencing data through final characterization. It performs high-quality hybrid assembly using Autocycler (with Dragonflye fallback if Autocycler fails), followed by Medaka polishing and downstream analysis and typing including gene annotation, antimicrobial resistance detection, and plasmid identification. 

*Note: as currently configured, it does not run QC steps but instead expects a QC file as input from a separate QC pipeline*

**Key Features:**
- **Smart Assembly Strategy**: Autocycler for optimal bacterial genome assembly with automatic Dragonflye fallback if Autocycler fails; also splits Medaka polishing to a separate step, to allow assembly (CPU-heavy) to run in higher parallel mode than Medaka (GPU-heavy) can handle
- **YAML-Configured**: Human-readable configuration makes adding new tools or changing parameters simple and easy to track exactly what commands are being run
- **Resource Monitoring**: Built-in per-tool/per-sample CPU/memory tracking for accurate time and resource requirement tracking, with optimization suggestions
- **Robust Execution**: Automatic output detection, dependency management, and error handling
- **Robust Recovery**: Re-run will skip existing output (unless using --force mode), for easy recovery from failure
- **Single Tool Runs**: Can run just a single tool at a time
- **Version Tracking**: Tracks version numbers and script/database paths for reproducibility
- **Comprehensive Logging**: Produces detailed full-pipeline and per-sample run logging for traceability and error recovery
- **HPC Ready**: Module system integration and configurable parallelism for high-performance computing

## Quick Start

```bash
# Load python module for required python packages
module load python

# Run complete pipeline for a normal sequencing run
python3 bactopipe.py -r RUNID

# Run with specific options
python3 bactopipe.py -r RUNID --force --clean

# Run specific tools only
python3 bactopipe.py -r RUNID --tools prokka abritamr mlst

# Check tool versions without running
python3 bactopipe.py --tool_versions

# Direct input/output mode
python3 bactopipe.py -i samples.tsv -o output_directory/
```

## Command Line Options

### Required Arguments (one of):
- `-r, --runid RUNID` - Run ID for ONT sequencing run (e.g., 20241118_RDRD)
- `-i, --input FILE` and `-o, --output DIR` - Custom input samples file and output directory

### Optional Flags:
- `--force` - Overwrite existing outputs (default: skip completed analyses)
- `--dry-run` - Show commands without executing them
- `--skip` - Continue processing even if some tools fail
- `--clean` - Start fresh log files instead of appending
- `--tools TOOL [TOOL...]` - Run only specified tools (e.g., `--tools prokka mlst`)
- `--tool_versions` - Display versions of all configured tools and exit
- `-c, --config FILE` - Use alternate config file (default: bactopipe_config.yaml)
- `-v, --version` - Report pipeline version number

## Input Requirements

### Sample File Format
The pipeline expects a tab-separated file (`samples.tsv`) with the following structure:

```
BARCODE	SAMPLE_ID	BARCODE_NAME	EXPECTED_SPECIES	IDENTIFIED_SPECIES	IDENTIFIED_GENOME_SIZE
barcode01	SAMPLE001	BC01	Escherichia coli	Escherichia coli	5000000
barcode02	SAMPLE002	BC02	Salmonella enterica	Salmiella enterica	4800000
barcode03	NEG	BC03	negative_control	Unclassified	0
```

**Core Requirements:**
- **Pipeline core** requires only a column named `SAMPLE_ID` (configurable via `sample_id_column` setting)
- **Header line** is required for column name detection
- **Current tool configuration** expects the full 6-column format shown above

**Column descriptions:**
- `BARCODE` - ONT barcode identifier (used by summarise_run.py)
- `SAMPLE_ID` - **Required**: Unique sample identifier used throughout pipeline
- `BARCODE_NAME` - Short barcode name (used by summarise_run.py)
- `EXPECTED_SPECIES` - Expected species (used by summarise_run.py)
- `IDENTIFIED_SPECIES` - Species identified by taxonomic classification (used by summarise_run.py)
- `IDENTIFIED_GENOME_SIZE` - Estimated genome size in base pairs (used by run_assembly.sh for optimization)

**Notes:**
- Header line is required and will be skipped during processing
- Samples with `SAMPLE_ID` matching entries in `allow_failed_sample_ids` (config setting) will be allowed to fail dependency checks
- Empty or `0` values in `IDENTIFIED_GENOME_SIZE` will trigger meta-assembly mode in some tools

## Pipeline Workflow 
### (Per bactopipe_config.yaml)

```mermaid
%%{init: {
  'flowchart': {
    'nodeSpacing': 30,
    'rankSpacing': 50,
    'curve': 'basis'
  }
}}%%
flowchart TD
    A[Input: samples.tsv] --> B[Pipeline Start]
    B --> C[Assembly: Autocycler /<br/> Dragonflye fallback]
    C --> D[Medaka <br/>Assembly Polishing]
    
    D --> E[Prokka <br/>rough ORF annotation]
    D --> F[MLST <br/>Sequence Typing]
    D --> G[AbriTAMR / AMRFinder <br/>AMR prediction]
    D --> H[CheckM <br/>assembly QC]
    D --> I[PlasmidFinder <br/>replicon detection]

    E --> J[PADLOC </br> phage defense systems]
    I --> K[parse_plasmids.py <br/> identify plasmid contigs]

    E --> L[run_summary.py <br/> Generate Summary]
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L

    B --> N[Resource Monitoring]
    
    L --> P[summary.tsv]
    K --> Q[plasmid_contigs.tsv]
    B --> R[versions.tsv]
    N --> O[resource_usage.tsv]

    style A fill:#e8f5e8
    style C fill:#fce4ec
    style D fill:#fce4ec
    style E fill:#e1f5fe
    style J fill:#e1f5fe
    style L fill:#f3e5f5
    style P fill:#e8f5e8
    style Q fill:#e8f5e8
    style R fill:#e8f5e8
    style N fill:#fff2cc
    style O fill:#fff2cc
```

## Expected Output Structure

```
/data/runs/ont/analysis/RUNID/
├── RUNID.run.log                # Main pipeline execution log
├── RUNID.versions.TIMESTAMP.tsv # Tool versions and databases used
├── RUNID.summary.tsv            # Final summary table (all results)
├── RUNID.plasmid_contigs.tsv    # Plasmid-AMR contig analysis
├── samples.tsv                  # Sample manifest (copied from QC)
├── assembly/                    # Genome assemblies
│   ├── {sample_id}.fa              # Final polished assemblies
│   ├── {sample_id}.unpolished.fa   # Raw assemblies
│   ├── autocycler/{sample_id}/     # Autocycler output
│   ├── dragonflye/{sample_id}/     # Dragonflye output (fallback)
│   ├── medaka/{sample_id}/         # Medaka polishing intermediates
│   │   ├── round1/                 # First polishing round
│   │   ├── round2/                 # Second polishing round  
│   │   └── {sample_id}.polished.fa # Final polished output
│   └── logs/                       # Consolidated assembly logs
├── prokka/                      # Prokka annotations
│   └── {sample_id}/                # Per-sample annotation files (.gff, .faa, etc.)
├── abritamr/                    # AMR analysis
│   ├── abritamr.txt                # Combined results
│   └── {sample_id}/                # Per-sample detailed results
├── mlst/                        # MLST typing
│   └── mlst.csv                    # Combined results
├── plasmidfinder/               # Plasmid replicon identification
│   ├── plasmidfinder_results.tsv  # Combined results
│   └── {sample_id}/                # Per-sample detailed results
├── checkm/                      # Assembly QC metrics
│   ├── checkm_results.tsv         # Combined results
│   └── {sample_id}/                # Per-sample detailed results
└── padloc/                      # Phage defense systems
    ├── padloc_summary.tsv          # Combined results
    └── {sample_id}/                # Per-sample PADLOC output
```

## Configuration Guide

The pipeline is controlled by `bactopipe_config.yaml`. Each tool is defined with specific parameters controlling execution mode, parallelism, and dependencies.

### Tool Configuration Structure

```yaml
tools:
  tool_name:
    description: "Tool purpose"
    modules: [module/version]          # Environment modules to load
    execution_mode: batch              # 'batch' or 'per_sample'
    parallel: 4                        # Concurrent jobs (per_sample mode only, 1=sequential)
    output_dir: "{rundir}/tool"        # Where outputs go
    output_file: "{output_dir}/final.tsv"        # Expected output (skip check)
    sample_output_file: "{output_dir}/{sample_id}.txt"  # Per-sample output (skip check)
    command: |                         # Main command to execute
      command(s) to execute
    pre_commands: []                   # Setup commands (optional)
    post_commands: []                  # Cleanup commands (optional)
    version_cmd: "tool --version"      # Version detection (optional)
    tool_path: "/path/to/tool"         # Tool location (optional)
    db_path: "/path/to/database"       # Database location (optional)
    db_cmd: "command to get db path"   # Dynamic database detection (optional)
```

### Execution Modes

**Batch mode**: Runs once for all samples together
- Tool executes: pre_commands → command → post_commands
- Single process handles entire dataset
- Example: MLST typing all genomes in one command

**Per-sample mode**: Runs separately for each sample
- Tool executes: pre_commands → parallel(command per sample) → post_commands
- `parallel` controls concurrency (default: 4, set to 1 for sequential)
- Thread count applies to sample parallelism, not tool threads
- Example: Prokka annotating each genome individually

### Variable Substitution

Variables replaced at runtime:
- `{runid}` - Run identifier
- `{rundir}` - Analysis output directory path
- `{sample_id}` - Current sample id (per_sample mode only)
- `{assembly_file}` - Path to sample assembly file
- `{script_dir}` - Pipeline installation directory
- `{any_tool_field}` - Any field from the tool's config block

### Skip Detection

Pipeline checks outputs before running tools:

1. For **batch** tools:
   - First checks `output_file` if defined
   - Then checks `sample_output_file` pattern for all samples
   - Skips if all expected outputs exist

2. For **per_sample** tools:
   - Checks `output_file` for final combined output
   - Checks each sample's `sample_output_file`
   - Skips individual samples with existing outputs
   - Runs only missing samples

Use `--force` to override skip detection.

### Version and Database Tracking

Version information is collected for the `.versions.tsv` file:

#### Version detection:
The pipeline calls `{tool_name} --version` by default. If this won't work for your tool,
you can specify a custom command using:
```
version_cmd: "command"  # Custom command to extract version info if needed
```

#### Tool path detection:
The pipeline calls `which {tool_name}` by default. If you need to specify an exact path,
you can set:
```
tool_path: "/path/to/tool"  # Explicit tool path if needed
```
#### Database detection:
The pipeline reports "none" by default. If you want to track database locations,
you can specify either:
```
db_path: "/data/db/tool_db"   # Static database path
db_cmd: "command"             # Custom command to extract database info from tool/logs
```

### Complete Example

Adding a new AMR tool:

```yaml
tools:
  resfinder:
    description: "ResFinder antimicrobial resistance gene detection"
    modules: [resfinder/4.1.0]
    execution_mode: per_sample
    parallel: 8                        # Run 8 samples simultaneously
    output_dir: "{rundir}/resfinder"
    sample_output_file: "{output_dir}/{sample_id}/results.json"
    pre_commands:
      - mkdir -p {output_dir}
    command: |
      resfinder.py \
        -i {assembly_file} \
        -o {output_dir}/{sample_id} \
        -s "Escherichia coli" \
        --acquired
    post_commands:
      - python3 {script_dir}/merge_resfinder.py {output_dir}
    version_cmd: "resfinder.py --version 2>&1 | head -1"
    db_path: "/opt/resfinder/database"
```

This configuration will:
1. Check if sample outputs exist before running
2. Create output directory once
3. Run ResFinder on up to 8 samples in parallel
4. Merge results after all samples complete
5. Track version and database in pipeline logs


## Installation and Dependencies

### System Requirements
The pipeline uses a module system for dependency management where possible but can also work on straight install paths.

### Environment Modules
Environment modules are a Linux system for dynamically loading different versions of software tools without conflicts. Each tool is installed in its own conda environment and loaded via the module system. When you load a module, it adds the tool to your PATH and sets up the required environment.

**Learn more:** [Environment Modules User Guide](https://modules.readthedocs.io/en/latest/module.html)

### Module System in the Pipeline
Tools are automatically loaded using environment modules as specified in `bactopipe_config.yaml`. The pipeline handles all module loading - you don't need to manually load modules before running.

```yaml
modules:
  - prokka/1.14.6
```

### Available Tool Modules
- `assembly` - Genome assembly (via autocycler/dragonflye)
- `prokka/1.14.6` - Genome annotation
- `abritamr/1.0.17` - AMR detection with AMRFinderPlus
- `mlst/2.23.0` - Multi-locus sequence typing
- `plasmidfinder/2.1.6` - Plasmid identification
- `checkm-genome/1.2.2` - Assembly quality control
- `padloc/2.0.0` - Defense system detection

### Python Dependencies
- Python 3.8+
- PyYAML
- Standard library only (pathlib, subprocess, concurrent.futures)

### Database Locations
Databases are managed per-tool and tracked in the versions output:
- ABRitAMR: AMRFinderPlus database (auto-updated)
- PlasmidFinder: Plasmid reference database
- CheckM: Marker gene database
- PADLOC: HMM profiles for defense systems

### Setting Up a New Installation
1. Ensure all tool modules are available:
   ```bash
   module avail 2>&1 | grep -E "prokka|abritamr|mlst|plasmidfinder|checkm|padloc"
   ```

2. Clone the pipeline:
   ```bash
   git clone /data/projects/pipelines/dev/jennyd/analysis_v2
   ```

3. Verify configuration:
   ```bash
   python3 bactopipe.py --tool_versions
   ```

4. Test with a small dataset:
   ```bash
   python3 bactopipe.py -r TEST_RUN --dry-run
   ```

## Resource Management

BactoPipe includes intelligent resource monitoring and optimization:

- **Automatic monitoring** of CPU and memory usage per tool
- **Resource-aware parallelism** with bottleneck detection  
- **Optimization suggestions** for improving performance
- **Configurable limits** with safety caps to prevent system overload

Resource logs are saved to `{runid}.resource_usage.{timestamp}.tsv` for analysis.

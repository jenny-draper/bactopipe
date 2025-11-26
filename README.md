# BactoPipe - A Lightweight, YAML-Configured Pipeline Framework

**A "workflow-light" python runner that brings workflow system features to your existing scripts and tools.**

## What is BactoPipe?

BactoPipe is a pipeline framework that sits between simple bash scripts and full workflow management systems. It is designed primarily for microbiolal genomics laboratories (and others) with existing in-house bash/python pipelines running on a locally-administered Linux server, who want workflow-like features without the conceptual overhead of a full workflow language.

**BactoPipe is designed for users who:**
- Have existing scripts and ad-hoc pipelines they want to organize
- Want workflow system features without learning a workflow language
- Work in regulated environments requiring detailed logging and audit trails
- Want to easily add, remove, or update analysis tools
- Prefer growing their own pipeline over adopting turnkey solutions

Think of it as a way to transform your collection of scripts and commands into a well-organized, traceable, and human-readable pipeline with minimal effort (an AI can likely do most of the conversion for you!).

## Core Concept

**your tools & scripts → pipeline.yaml → bactopipe.py runner → output**

1. **Define your pipeline in human-readable YAML**, including bash commands, dependencies, and per-tool parallelism:

```yaml
# my_pipeline.yaml
tools:
  my_tool:
    command: "my_tool {input_file} {output_dir}"                   # bash command to run tool
    parallel: 4                                                    # run 4 samples from input file at once
    sample_output_file: "{output_dir}/{sample_id}.result.txt"      # presence = successful run
```

2. **The bactopipe.py runner handles execution**, standardising input/output handling and producing detailed, organised logs:

```bash
python bactopipe.py -c my_pipeline.yaml -i input_file -o /path/to/output
```

## Key Features

### 🔄 Workflow-Like Execution
- **Smart resume**: Automatically skips completed steps (checks output files)
- **Dependency management**: Define tool order and requirements
- **Dry-run mode**: Validate pipeline before execution
- **Force re-run**: Override skip detection when needed

### 📊 Comprehensive Monitoring
- **Resource tracking**: CPU, memory, and runtime per tool/sample
- **Parallelism optimization**: Get suggestions based on actual usage
- **Version logging**: Automatic capture of tool versions and databases
- **Detailed audit trails**: Step-level and run-level logs for compliance

### 🔧 Flexible Integration
- **Stack-agnostic**: Use any tools - conda, modules, containers, or direct installs
- **Script-friendly**: Wrap existing bash/python scripts without modification
- **Variable substitution**: Dynamic paths and parameters
- **Multiple execution modes**: Batch processing or per-sample parallelism

### 🚀 Minimal Dependencies
- **Lightweight requirements**: Just Python 3.8+ with PyYAML and pandas
- **Built-in parallelization**: Uses Python's standard ThreadPoolExecutor
- **Optional monitoring**: GNU time for resource tracking (auto-disables if not found)
- **Environment flexibility**: Supports Linux modules, conda, or direct tool calls

## Demo

The `/demo` directory contains a `demo.yaml` file defining a complete minimal QC pipeline (FastQC + MultiQC) to demonstrate the basic concept and features, along with small input sample files for quick testing.

Given this `demo.yaml` file and the input/output file/directory paths to run on, `bactopipe.py `will run FastQC as described on 20 samples at a time, then run MulitQC to combine the results and generate a combined QC report. All of the output directory creation and structure, skip/resume/overwrite and logging features, etc are handled automatically by `bactopipe.py`!

*Note: this demo assumes you already have FastQC and MultiQC installed and in your PATH. If your system uses linux environment modules, edit `demo/demo.yaml` and uncomment/edit the `#modules: [module_name]` lines appropriately to load your tool modules.*


### Demo YAML Config
```yaml
# demo.yaml

settings:
  pipeline_name: "Demo minimal QC Pipeline"  
  version: 1.0

tools:
  fastqc:
    execution_mode: per_sample           
    parallel: 20
    output_dir: "{rundir}/fastqc"
    sample_output_dir: "{rundir}/fastqc/{sample_id}"
    sample_output_file: "{rundir}/fastqc/{sample_id}/{sample_id}*_fastqc.html"  
    command: "fastqc -o {sample_output_dir} {input_dir}/{sample_id}*.fastq.gz"

  multiqc:
    execution_mode: batch
    output_file: "{rundir}/multiqc_report.html" #if defined outputs exist already, will skip
    command: "multiqc {rundir}/fastqc -o {rundir}"

dependencies:
  multiqc: [fastqc]
```

### To Run the Demo

```bash
# Clone the repository
git clone https://github.com/jenny-draper/bactopipe
cd bactopipe

# Run demo
python3 bactopipe.py -c demo/demo.yaml -i demo/samples.tsv --input-dir demo/fastqs --output-dir demo/qc
```

### Example Output

```
$ python bactopipe.py -c demo/demo.yaml -i demo/samples.tsv --input-dir demo/fastqs --output-dir demo/qc

============================================================
BactoPipe v1.1: Demo minimal QC Pipeline
============================================================
Command: python3 bactopipe.py -c demo.yaml -i samples.tsv -o demo/qc
Config file: demo/demo.yaml
Output directory: /home/user/bactopipe/demo/qc
Target samples file: demo/samples.tsv
System Resources: 96 CPU cores (192 logical cores), 1007.4 GB RAM
System Current Load: Memory: 35.5/1007.4 GB (3.5%), CPU (5min avg): 0.12 (0.1%)
Flags: None
User: username
Timestamp: 2025-01-15 14:23:10
============================================================

Tools to run: ['fastqc', 'multiqc']

=== Running fastqc ==========================================
Logging run to: /home/user/bactopipe/demo/qc/fastqc/fastqc.log
Logging per-sample tool output to: /home/user/bactopipe/demo/qc/fastqc/*/*.fastqc.log
Run mode: 20 samples in parallel
⚠️  fastqc: missing outputs for 2 samples: sample01, sample02
fastqc Start: 2025-01-15 14:23:10
sample01: Starting
sample02: Starting
sample01: Completed (runtime: 18.3 seconds)
sample02: Completed (runtime: 29.3 seconds)
✅ fastqc completed successfully
fastqc End | Total Runtime: 29.3 seconds | Peak Memory: 0.63 GB | CPU: 1.1 cores

=== Running multiqc =========================================
Logging run to: /home/user/bactopipe/demo/qc/logs/multiqc.log
Run mode: batch
multiqc Start: 2025-01-15 14:55:10
✅ multiqc completed successfully
multiqc End | Total Runtime: 1.8 seconds | Peak Memory: 0.10 GB | CPU: 9.8 cores

== Logging tool versions =====================================
Tool versions appended to: /home/jennyd/bactopipe/demo/qc/qc.versions.tsv
Resource usage appended to: /home/jennyd/bactopipe/demo/qc/qc.resource_usage.tsv

==================================================
Pipeline completed for qc - all tools
Total runtime: 32.2 seconds
Timestamp: 2025-11-12 13:57:17
```

### Output

```
demo/qc/
├── fastqc/               # FastQC run log & per-sample output
│   └── sample01/         # FastQC output and log for sample01  
├── multiqc_data/         # MultiQC output report data
├── multiqc_report.html   # MultiQC output report
├── logs/                 # Logs for tools without specified output directories (e.g. MultiQC)
├── run.log               # Execution log
├── resource_usage.tsv    # Performance metrics
└── versions.tsv          # Tool versions
```

### Resource Usage and Version Tracking

**Resource Usage: (output file `resource_usage.tsv`)**
*(this is a subset of columns)
```
timestamp            tool     n_ran   mean_memory_gb  mean_cpu  max_cpu  runtime_sec  suggested_threads  bottleneck
2025-01-15 14:23:45  fastqc   2       0.617           1.1       1.1      29.3         135                cpu
2025-01-15 14:24:16  multiqc  1       0.103           9.8       9.8      1.8          15                 cpu
```

**Version Tracking: (output file `versions.tsv`)**
```
timestamp            tool                    version              path                           database
2025-01-15 14:23:10  bactopipe.py           1.1                  /home/user/bactopipe.py       none
2025-01-15 14:23:10  bactopipe_config.yaml  1.0                  /home/user/demo.yaml          none
2025-01-15 14:24:16  fastqc                 FastQC v0.12.1       /opt/conda/bin/fastqc         none
2025-01-15 14:24:16  multiqc                multiqc, version 1.21 /opt/conda/bin/multiqc       none
```

## Command-Line Options

```bash
# Process a run with default settings
python bactopipe.py -c pipeline.yaml -i input_file -o /path/to/output
python bactopipe.py -c pipeline.yaml --input-dir /path/to/raw -o /path/to/output

# Resume from where you left off (automatically skips completed steps!)
python bactopipe.py -c pipeline.yaml --input-dir /path/to/raw -o /path/to/output

# Run specific tools only
python bactopipe.py ... --tools kraken2 summarise

# Force re-run 
python bactopipe.py ... --force

# Allow specific samples to fail without stopping pipeline
python bactopipe.py ... --skip NEG,sample001

# Preview commands without execution
python bactopipe.py ... --dry-run

# Show more detail in command-line output
python bactopipe.py ... --verbose
```


## Command Line Options Reference

| Option | Description |
|--------|-------------|
| `-c, --config` | YAML configuration file (required) |
| `-i, --input` | Input samples file (TSV with SAMPLE_ID column) |
| `--input-dir` | Input directory (alternative input type) |
| `-o, --output` | Output directory (required) |
| `--tools` | Run specific tools only |
| `--force` | Force re-run even if outputs exist |
| `--dry-run` | Show commands without executing |
| `--skip` | Comma-separated list of samples to allow to fail |
| `--verbose` | Print commands to console |
| `--clean` | Start fresh logs (don't append) |

## Configuration Reference

See [`config_template.yaml`](config_template.yaml) for a complete reference of all YAML configuration options.

## Examples

- **Demo QC pipeline**: [`demo/demo.yaml`](demo/demo.yaml) - Minimal FastQC + MultiQC example
- **Illumina QC pipeline**: [`qc_illumina/qc_illumina.yaml`](qc_illumina/amr_qc.yaml) - a full bacterial Illumina QC pipeline including read trimming, subsampling, species identifcation, and run summary file generation using a combination of standard tools and custom scripts.


## Documentation

*More detailed documention is on its way!*

## Citation

If you use BactoPipe,fplease cite this repository:
https://github.com/jenny-draper/bactopipe


## Comparison with Other Approaches

| Feature | Bash Scripts | BactoPipe | Workflow Systems |
|---------|-------------|-----------|------------------|
| Learning curve | Low | Low-Medium | High |
| Parallelism | Manual | Automatic | Automatic |
| Resume capability | Manual | Automatic | Automatic |
| Resource monitoring | No | Yes | Yes |
| Version tracking | Manual | Automatic | Varies |
| Audit logging | Manual | Automatic | Automatic |
| Dependencies | Manual | YAML-defined | DSL-defined |
| Infrastructure needs | None | None | Often complex |



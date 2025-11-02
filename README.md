# BactoPipe - A Lightweight, YAML-Configured Pipeline Framework

**A "workflow-light" Python runner that brings workflow system features to your existing scripts and tools - without the complexity.**

## What is BactoPipe?

BactoPipe is a pipeline framework that sits inbetween simple bash scripts and full workflow management systems. It's designed for microbiology groups (and others) who need:

- **Reproducible, parallel analyses** on a single Linux box
- **No workflow language or job scheduler** to learn
- **Detailed audit trails** for clinical/accredited settings
- **Easy updates and targeted re-runs** of specific tools
- **Clear provenance and execution telemetry** without conceptual overhead

Think of it as a way to transform your collection of scripts and commands into a well-organized, traceable pipeline with minimal effort.

## Key Features

### 🎯 Simple YAML Configuration
Define your pipeline in human-readable YAML - just list your tools, commands, and dependencies:
```yaml
tools:
  my_tool:
    command: "my_tool {input_file} {output_dir}"
    parallel: 4  # Run 4 samples from input file at once
    output_file: "{output_dir}/result_file"
```

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

## Who is BactoPipe For?

BactoPipe targets users who:
- Have existing scripts and ad-hoc pipelines they want to organize
- Need workflow system features without learning a workflow language
- Work in regulated environments requiring detailed logging
- Want to easily add, remove, or update analysis tools
- Prefer growing their own pipeline over adopting turnkey solutions

## Quick Start

1. **Create a YAML config** describing your tools:
```yaml
settings:
  pipeline_name: "My Analysis Pipeline"
  default_parallel: 4

tools:
  step1:
    command: "my_analysis.sh {sample_id}"
    output_file: "{rundir}/results/{sample_id}.txt"
  
  step2:
    command: "summarize.py {rundir}/results"
    output_file: "{rundir}/summary.tsv"

dependencies:
  step2: [step1]  # step2 requires step1 to complete first
```

2. **Run your pipeline**:
```bash
# Process samples
python3 bactopipe.py -i samples.tsv -o output_dir/

# Or with run ID mode
python3 bactopipe.py -r RUN_ID

# Dry run to see what would execute
python3 bactopipe.py -r RUN_ID --dry-run

# Re-run specific tools only
python3 bactopipe.py -r RUN_ID --tools step2 --force
```

3. **Get comprehensive outputs**:
```
output_dir/
├── pipeline.run.log           # Complete execution log
├── versions.tsv               # Tool versions and paths
├── resource_usage.tsv         # Performance metrics
└── [your tool outputs]        # As defined in YAML
```

## Core Concepts

### Execution Modes
- **Batch**: Run once for all samples
- **Per-sample**: Parallel processing with configurable threads

### Skip Detection
BactoPipe checks for existing outputs before running tools. Define `output_file` or `sample_output_file` patterns to enable automatic skipping of completed work.

### Resource Optimization
The framework monitors resource usage and suggests optimal parallelism:
```
tool      max_memory_gb  max_cpu_cores  current_threads  suggested_threads
my_tool   3.8           2.5            4                8
```

### Version Tracking
Automatic logging of tool versions, paths, and databases for full reproducibility:
```
tool         version    path                    database
my_tool      v1.2.3     /usr/local/bin/tool    /data/db/v2024
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input samples file (TSV with SAMPLE_ID column) |
| `-o, --output` | Output directory |
| `-r, --runid` | Run ID (alternative mode with predefined paths) |
| `--tools` | Run specific tools only |
| `--force` | Force re-run even if outputs exist |
| `--dry-run` | Show commands without executing |
| `--skip` | Skip tools with missing dependencies |
| `--verbose` | Print commands to console |
| `--clean` | Start fresh logs (don't append) |

## Examples

### Example 1: Simple QC Pipeline
```yaml
tools:
  fastqc:
    command: "fastqc {input_file} -o {output_dir}/fastqc"
    parallel: 8
    execution_mode: per_sample
    
  multiqc:
    command: "multiqc {output_dir}/fastqc -o {output_dir}"
    execution_mode: batch
    
dependencies:
  multiqc: [fastqc]
```

### Example 2: Assembly Pipeline
See [`bactopipe_config.yaml`](bactopipe_config.yaml) for a complete bacterial genome analysis pipeline example with assembly, annotation, AMR detection, and more.

## Installation

### Requirements
- Python 3.8+
- PyYAML (`pip install pyyaml`)
- pandas (`pip install pandas`)

### Setup
```bash
# Clone repository
git clone https://github.com/jenny-draper/bactopipe
cd bactopipe

# Install Python dependencies
pip install pyyaml pandas

# Test with example config
python3 bactopipe.py --dry-run -i samples.tsv -o test_output/
```

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

## Documentation

- [YAML Configuration Guide](docs/yaml_guide.md) - Detailed configuration options
- [Adding New Tools](docs/adding_tools.md) - How to integrate your tools
- [Module System Guide](docs/modules.md) - Using environment modules
- [Example Pipelines](examples/) - Sample configurations for common analyses

## Citation

If you use BactoPipe in your research, please cite:
```
[Citation information to be added upon publication]
```

## License

[License information]

## Support

- Issues: [GitHub Issues](https://github.com/jenny-draper/bactopipe/issues)
- Documentation: [GitHub Wiki](https://github.com/jenny-draper/bactopipe/wiki)

---

*BactoPipe: Bringing workflow system features to your scripts, without the workflow system.*
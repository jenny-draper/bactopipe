#!/usr/bin/env python3
"""
BactoPipe - Bacterial Sequence Analysis Pipeline
==============================================================================

A YAML-configurable pipeline framework for bacterial genome analysis.
Features parallel processing, resource monitoring, and 
intelligent dependency management.

Author: Jenny L Draper
Date: October 2025

Usage:
    bactopipe.py -i samples.tsv -o output_dir/    # Process samples
    bactopipe.py --help                           # Show full help
"""

import argparse
import datetime
import glob
import os
import shlex
import subprocess
import sys
import time
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple

VERSION = "1.1"

# Default values for pipeline behavior
DEFAULTS = {
    'fallback_runid': 'pipeline_run',
    'resource_utilization_target': 0.8,
    'default_version_cmd_pattern': '{tool_name} --version',
    'default_database_value': 'none',
    'default_unknown_value': 'unknown',
    'sample_id_column_name': 'SAMPLE_ID',
    'subprocess_timeout': 10,
    'pipeline_log_pattern': '{runid}.run.log',
    'sample_log_pattern': '{sample_id}.{tool_name}.log',
    'tool_log_pattern': '{tool_name}.log',
    'default_execution_mode': 'per_sample',
    'default_parallel': 1
}

class PipelineRunner:
    def __init__(self, config_file: str, input_file: str = None, input_dir: str = None, 
                 output_dir: str = None, runid: str = None, dry_run: bool = False, force: bool = False, skip_samples: str = None, 
                 clean: bool = False, monitor: bool = True, verbose: bool = False):
    
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate config was loaded successfully
        if self.config is None:
            raise ValueError(f"Config file is empty or invalid YAML: {config_file}")
        if not isinstance(self.config, dict):
            raise ValueError(f"Config file must contain a YAML dictionary/mapping: {config_file}")
        
        # Merge skip_samples with config's allow_failed_sample_ids
        if skip_samples:
            # Parse comma-separated list
            parsed_samples = [s.strip() for s in skip_samples.split(',') if s.strip()]
            if parsed_samples:
                # Merge with existing config
                self.config.setdefault('settings', {}).setdefault('allow_failed_sample_ids', []).extend(parsed_samples)
        
        self.config_file = config_file
        
        # Set script_dir from config if specified, otherwise use config file location
        config_script_dir = self.config.get('settings', {}).get('script_dir')
        if config_script_dir:
            self.script_dir = Path(config_script_dir).absolute()
        else:
            # Default to directory containing the config file
            self.script_dir = Path(config_file).parent.absolute()
        
        self.monitor = monitor
        
        # setup output directory and runid 
        if not output_dir:
            raise ValueError("output_dir is required")
        self.rundir = Path(output_dir).absolute()  # Make absolute
        
        # Use provided runid or empty string
        self.runid = runid if runid else ""
        
        # Determine input file location
        if input_file:
            # Direct mode with samples file
            self.input_file = Path(input_file)
        else:
            # Use configured default sample file path
            default_file = self.config.get('settings', {}).get('default_sample_file')
            if not default_file:
                raise ValueError("default_sample_file must be set in config when --input-file is not provided")
            default_path = default_file.replace('{rundir}', str(self.rundir))
            self.input_file = Path(default_path)

        # Convert input_dir to absolute path if provided
        self.input_dir = str(Path(input_dir).resolve()) if input_dir else None

        # handle the remaining flags / settings  
        self.dry_run = dry_run
        self.force = force
        self.clean = clean
        self.verbose = verbose
        self.current_log_file = None
        self.pipeline_log_file = None
        self.resource_data = []
        self.system_resources = self.get_system_resources()
        self.executed_tools = []  # Track which tools actually ran
        self.pipeline_start_time = None  # Track pipeline start time

    def get_timestamp(self) -> str:
        """Get current timestamp in standard format."""
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def init_log_file(self, log_file: Path, log_name: str) -> None:
        """Initialize or append to a log file with consistent formatting."""
        mode = 'w' if self.clean else 'a'
        with open(log_file, mode) as f:
            if not self.clean and log_file.exists() and log_file.stat().st_size > 0:
                f.write(f"\n\n\n# {log_name} started at {self.get_timestamp()} {'=' * 30}\n")
            else:
                f.write(f"# {log_name} started at {self.get_timestamp()} {'=' * 30}\n")

    def setup_pipeline_logging(self):
        """Set up logging for the entire pipeline run."""
        # Use consistent pattern: if runid exists, use it; otherwise use generic name
        log_filename = f"{self.runid}.run.log" if self.runid else "run.log"
        self.pipeline_log_file = self.rundir / log_filename
        self.init_log_file(self.pipeline_log_file, "Pipeline run log")

    def write_to_file(self, message: str, log_file: Path, prefix: str = "") -> None:
        """Write message to a specific log file with optional prefix."""
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{prefix}{message}\n")

    def log(self, message: str, console: bool = True, tool_log: bool = True, sample_log: Path = None) -> None:
        """Unified logging method that handles all logging scenarios."""
        # Console and pipeline log
        if console:
            print(message)
            if self.pipeline_log_file:
                self.write_to_file(message, self.pipeline_log_file)
        
        # Tool log (with Bactopipe: prefix)
        if tool_log and self.current_log_file:
            self.write_to_file(message, self.current_log_file, prefix="Bactopipe: ")
        
        # Sample log (with Bactopipe: prefix)
        if sample_log:
            self.write_to_file(message, sample_log, prefix="Bactopipe: ")

    def print_startup_info(self):
        """Print startup banner with key information."""
        self.pipeline_start_time = time.time()
        pipeline_name = self.config.get('settings', {}).get('pipeline_name', 'Pipeline')
        self.log("=" * 60)
        self.log(f"BactoPipe v{VERSION}: {pipeline_name}")
        self.log("=" * 60)
        self.log(f"Command: {' '.join(sys.argv)}")
        if self.runid:  # Only show Run ID line if it's set
            self.log(f"Run ID: {self.runid}")
        self.log(f"Config file: {self.config_file}")
        self.log(f"Output directory: {self.rundir}")
        self.log(f"Target samples file: {self.input_file}")
        
        if self.monitor:
            cpu_info = self._format_cpu_info()
            
            # Format memory information
            mem_total = self.system_resources.get('memory_gb', 'unknown')
            mem_used = self.system_resources.get('memory_used_gb', 'unknown')
            
            mem_info = f"{mem_total} GB RAM" if mem_total != 'unknown' else "RAM: unknown"
            
            # System Resources line (capacity)
            self.log(f"System Resources: {cpu_info}, {mem_info}")
            
            # System Current Load line (memory usage and 5-min CPU load average)
            load_5 = self.system_resources.get('cpu_load_5min', 'unknown')
            logical_cores = self.system_resources.get('logical_cores', 'unknown')
            
            usage_parts = []
            if mem_total != 'unknown' and mem_used != 'unknown':
                mem_percent = round((mem_used / mem_total) * 100, 1)
                usage_parts.append(f"Memory: {mem_used}/{mem_total} GB ({mem_percent}%)")
            
            if load_5 != 'unknown' and logical_cores != 'unknown':
                load_percent_5 = round((load_5 / logical_cores) * 100, 1)
                usage_parts.append(f"CPU (5min avg): {load_5} ({load_percent_5}%)")
            elif load_5 != 'unknown':
                usage_parts.append(f"CPU (5min avg): {load_5}")
            
            if usage_parts:
                self.log(f"System Current Load: {', '.join(usage_parts)}")
        
        settings = []
        if self.dry_run: settings.append("DRY RUN")
        if self.force: settings.append("FORCE")
        
        # Show allowed failures if any exist
        allowed_failures = self.config.get('settings', {}).get('allow_failed_sample_ids', [])
        if allowed_failures:
            settings.append(f"Allowing these samples to fail: {', '.join(allowed_failures)}")
        
        if not self.monitor: settings.append("NO MONITORING")
        
        self.log(f"Flags: {', '.join(settings) if settings else 'None'}")
        self.log(f"User: {os.getenv('USER', 'unknown')}")
        self.log(f"Timestamp: {self.get_timestamp()}")
        self.log("=" * 60)

    def _format_cpu_info(self) -> str:
        """Format CPU information string."""
        cpu = self.system_resources['cpu_cores']
        logical = self.system_resources['logical_cores']
        
        if cpu != 'unknown' and logical != 'unknown' and cpu != logical:
            return f"{cpu} CPU cores ({logical} logical cores)"
        elif logical != 'unknown':
            return f"{logical} CPU cores"
        else:
            return "CPU cores: unknown"

    def get_sample_ids(self) -> List[str]:
        """Extract sample IDs from the samples.tsv file using column name."""
        import pandas as pd
        
        try:
            # Read the TSV file
            df = pd.read_csv(self.input_file, sep='\t')
            
            # Get the column name to use (configurable, defaults to 'SAMPLE_ID')
            column_name = self.config.get('settings', {}).get('sample_id_column', DEFAULTS['sample_id_column_name'])
            
            # Check if the column exists
            if column_name not in df.columns:
                available_columns = ', '.join(df.columns.tolist())
                raise ValueError(f"Column '{column_name}' not found in {self.input_file}. Available columns: {available_columns}")
            
            # Extract sample IDs and return as list
            sample_ids = df[column_name].dropna().astype(str).tolist()
            
            if not sample_ids:
                raise ValueError(f"No sample IDs found in column '{column_name}' of {self.input_file}")
            
            return sample_ids
            
        except Exception as e:
            # Provide concise error message for file reading issues
            column_name = self.config.get('settings', {}).get('sample_id_column', DEFAULTS['sample_id_column_name'])
            raise RuntimeError(f"Could not read samples file {self.input_file}: {e}\nExpected: tab-separated file with '{column_name}' column")

    def substitute_variables(self, text: str, sample_id: str = None, tool_config: Dict = None) -> str:
        """Substitute variables in text strings with actual values."""
        if not isinstance(text, str):
            return text
        
        # Standard substitutions
        substitutions = {
            'runid': self.runid,  # User-provided runid or fallback to rundir basename
            'rundir': str(self.rundir),
            'script_dir': str(self.script_dir),
            'sample_id': sample_id or '',
            'assembly_file': ''
        }
        
        # Add input_dir if provided
        if hasattr(self, 'input_dir') and self.input_dir:
            substitutions['input_dir'] = str(self.input_dir)
        
        # Add all settings from config as substitutable variables
        for key, value in self.config.get('settings', {}).items():
            if isinstance(value, (str, int, float, bool)) and key not in substitutions:
                substitutions[key] = str(value)
        
        # Add assembly_file if needed
        if sample_id and 'default_assembly_path' in self.config.get('settings', {}):
            assembly_path = self.config['settings']['default_assembly_path']
            assembly_path = assembly_path.replace('{runid}', self.runid)
            assembly_path = assembly_path.replace('{rundir}', str(self.rundir))
            assembly_path = assembly_path.replace('{sample_id}', sample_id)
            substitutions['assembly_file'] = assembly_path
        
        # Add tool_config values if provided
        if tool_config:
            for key, value in tool_config.items():
                if isinstance(value, str) and '{' not in key:
                    substitutions[key] = self.substitute_variables(value, sample_id)
        
        # Do all substitutions
        result = text
        for key, value in substitutions.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        # NO GLOB EXPANSION - removed entirely
        
        return result

    def run_command(self, command: str, sample_id: str = None, log_file: Path = None, context: str = None) -> Tuple[float, float, float, int]:
        """Execute shell command and return resource usage plus return code."""
        full_command = self.substitute_variables(command, sample_id)
        
        prefix = f"{sample_id}: " if sample_id else ""
        
        if context:
            running_msg = f"{prefix}Running {context}:"
        else:
            running_msg = f"{prefix}Running:"
        
        # Format command with proper indentation
        formatted_command = '\n'.join('   ' + line for line in full_command.rstrip().split('\n'))
        
        # Only print commands to console if verbose flag is set
        self.log(f"{running_msg}\n{formatted_command}", console=self.verbose, tool_log=False)
        target_log_file = log_file or self.current_log_file
        if target_log_file:
            self.log(f"{prefix}Logging to: {target_log_file}", console=self.verbose, tool_log=False)
        
        # Log to tool-specific file with context
        log_msg = f"Bactopipe: Running {context}: {full_command}" if context else f"Bactopipe: Running: {full_command}"
        self.write_to_file(log_msg, target_log_file)
        
        if self.dry_run:
            return 0.0, 0.0, 0.0, 0
        
        time_stats_path = None
        needs_modules = hasattr(self, 'current_tool_modules') and self.current_tool_modules
        
        # Set up monitoring file if needed
        if self.monitor:
            if target_log_file:
                time_stats_path = str(target_log_file) + ".timer.tmp"
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as time_stats_file:
                    time_stats_path = time_stats_file.name
        
        # Build the command with appropriate wrappers
        if needs_modules:
            module_setup = " && ".join(["module purge"] + [f"module load {m}" for m in self.current_tool_modules])
            base_command = f"( . /etc/profile.d/modules.sh; {module_setup} && {full_command} )"
        else:
            base_command = full_command
        
        # Add monitoring wrapper if needed
        if self.monitor and time_stats_path:
            final_command = f'/usr/bin/time -v -o {time_stats_path} bash -c {shlex.quote(base_command)}'
        else:
            final_command = f'bash -c {shlex.quote(base_command)}'
        
        # Add output redirection if we have a log file
        if target_log_file:
            final_command = f'{final_command} >> {target_log_file} 2>&1'
            result = subprocess.run(final_command, shell=True, executable='/bin/bash')
            
            if result.returncode != 0:
                self.write_to_file(f"Bactopipe: Command exited with code {result.returncode}", target_log_file)
        else:
            result = subprocess.run(final_command, shell=True, executable='/bin/bash', capture_output=True, text=True)
            # Log captured output if not redirected
            if result.stdout and target_log_file:
                with open(target_log_file, 'a') as f:
                    f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr and target_log_file:
                with open(target_log_file, 'a') as f:
                    f.write(f"STDERR:\n{result.stderr}\n")
                    if result.returncode != 0:
                        f.write(f"EXIT CODE: {result.returncode}\n")
        
        # Parse resource usage if monitoring
        peak_memory_gb, cpu_cores, user_time = 0.0, 0.0, 0.0
        if self.monitor and time_stats_path:
            try:
                with open(time_stats_path, 'r') as f:
                    time_output = f.read()
                peak_memory_gb, cpu_cores, user_time = self.parse_time_output(time_output)
            except Exception:
                pass  # Keep defaults if reading fails
            finally:
                try:
                    os.unlink(time_stats_path)
                except Exception:
                    pass  # Ignore cleanup errors
        
        return peak_memory_gb, cpu_cores, user_time, result.returncode

    def setup_tool_logging(self, tool_name, run_dir):
        """Set up logging for tool-level operations (pre/post commands)"""
        tool_config = self.config['tools'].get(tool_name, {})
        
        # If tool specifies its own output_dir, put logs there
        if tool_config.get('output_dir'):
            log_dir = Path(self.substitute_variables(tool_config['output_dir']))
            log_dir.mkdir(parents=True, exist_ok=True)
        # Otherwise use shared logs directory
        else:
            log_dir = run_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
        
        log_filename = DEFAULTS['tool_log_pattern'].replace('{tool_name}', tool_name)
        log_file = log_dir / log_filename
        
        self.init_log_file(log_file, f"{tool_name} log")
        self.current_log_file = log_file
        return log_file

    @contextmanager
    def tool_logging_context(self, tool_name: str):
        """Context manager for tool-specific logging that ensures cleanup."""
        log_file = self.setup_tool_logging(tool_name, self.rundir)
        try:
            yield log_file
        finally:
            # Always clear the current log file when exiting tool context
            self.current_log_file = None

    def process_sample(self, tool_name: str, tool_config: Dict[str, Any], sample_id: str) -> Tuple[str, float, float, float]:
        """Execute tool on a single sample and return result with resource usage."""
        sample_log_file = None
        
        # Check if this sample is allowed to fail
        allow_failed_sample_ids = self.config.get('settings', {}).get('allow_failed_sample_ids', [])
        is_allowed_to_fail = sample_id in allow_failed_sample_ids
        
        try:
            # Check existing output first, before creating directories
            if not self.force:
                sample_output_file = tool_config.get('sample_output_file')
                if sample_output_file:
                    output_path = Path(self.substitute_variables(sample_output_file, sample_id))
                    if output_path.exists():
                        skip_message = f"✅ {sample_id} skipped (output exists: {output_path})"
                        self.log(skip_message, tool_log=True)
                        return f"{sample_id}: skipped (exists)", 0.0, 0.0, 0.0
            
            # Only create directories and set up logging if we're actually processing
            sample_output_dir = tool_config.get('sample_output_dir')
            if sample_output_dir:
                dir_path = Path(self.substitute_variables(sample_output_dir, sample_id))
                if not self.dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                log_filename = DEFAULTS['sample_log_pattern'].replace('{sample_id}', sample_id).replace('{tool_name}', tool_name)
                sample_log_file = dir_path / log_filename
                self.init_log_file(sample_log_file, f"{tool_name} log for {sample_id}")
                
                # Log directory creation - only to logs and verbose mode
                if not self.dry_run:
                    self.log(f"{sample_id}: Created directory {dir_path}", console=self.verbose, sample_log=sample_log_file)
            
            self.log(f"{sample_id}: Starting", sample_log=sample_log_file)
            
            # Start timing
            start_time = time.time()
            
            # Verify required files
            for file_pattern in tool_config.get('required_files', []):
                file_path = Path(self.substitute_variables(file_pattern, sample_id))
                if not file_path.exists():
                    self.log(f"{sample_id}: ERROR: missing input file {file_path}", sample_log=sample_log_file)
                    return f"{sample_id}: skipped", 0.0, 0.0, 0.0
            
            # Execute command
            command = self.substitute_variables(tool_config['command'], sample_id, tool_config)
            peak_memory_gb, cpu_cores, user_time, returncode = self.run_command(command, sample_id=sample_id, log_file=sample_log_file, context=f"{tool_name} command")
            
            # Check if command failed (unless sample is allowed to fail)
            if returncode != 0:
                if is_allowed_to_fail:
                    warning_msg = f"{sample_id}: Allowed failure (exit code {returncode})"
                    self.log(warning_msg, sample_log=sample_log_file)
                    return f"{sample_id}: allowed failure", 0.0, 0.0, 0.0
                else:
                    error_msg = f"{sample_id}: Failed with exit code {returncode}"
                    self.log(error_msg, sample_log=sample_log_file)
                    return f"{sample_id}: failed", 0.0, 0.0, 0.0
            
            # Calculate runtime and format completion message
            runtime = time.time() - start_time
            self.log(f"{sample_id}: Completed (runtime: {self.format_runtime(runtime)})", sample_log=sample_log_file)
            return f"{sample_id}: completed", peak_memory_gb, cpu_cores, user_time
            
        except Exception as e:
            if sample_log_file:
                self.log(f"{sample_id}: Failed - {e}", sample_log=sample_log_file)
            else:
                self.log(f"{sample_id}: Failed - {e}")
            return f"{sample_id}: failed", 0.0, 0.0, 0.0

    def check_dependencies(self, tool_name: str) -> Tuple[bool, List[str]]:
        """Check if all dependencies for a tool are satisfied. Returns (success, missing_details)."""
        dependencies = self.config.get('dependencies', {}).get(tool_name, [])
        missing_details = []
        
        for dep_tool in dependencies:
            if dep_tool not in self.config['tools']:
                continue
            
            dep_config = self.config['tools'][dep_tool]
            
            # Check if dependency tool has required outputs
            sample_output_file = dep_config.get('sample_output_file')
            output_file = dep_config.get('output_file')
            
            if sample_output_file:
                # Check per-sample outputs exist
                sample_ids = self.get_sample_ids()
                missing_samples, _ = self.check_sample_outputs(sample_output_file, sample_ids)
                
                # Filter out allowed sample IDs from missing samples
                allow_failed_sample_ids = self.config.get('settings', {}).get('allow_failed_sample_ids', [])
                filtered_missing = [s for s in missing_samples if s not in allow_failed_sample_ids]
                
                if filtered_missing:
                    missing_count = len(filtered_missing)
                    sample_list = ', '.join(filtered_missing[:3]) + ('...' if missing_count > 3 else '')
                    # Show the expected path pattern for the first missing sample
                    example_path = self.substitute_variables(sample_output_file, filtered_missing[0])
                    missing_details.append(f"{dep_tool} outputs missing for {missing_count} samples: {sample_list} (expected path: {example_path})")
                
                # Log if we're allowing some failures
                allowed_failures = [s for s in missing_samples if s in allow_failed_sample_ids]
                if allowed_failures:
                    allowed_list = ', '.join(allowed_failures[:3]) + ('...' if len(allowed_failures) > 3 else '')
                    self.log(f"ℹ️  {tool_name}: allowing specified samples with missing {dep_tool} dependencies: {allowed_list}")
            
            if output_file:
                # Check batch output exists
                output_path = Path(self.substitute_variables(output_file))
                if not output_path.exists():
                    missing_details.append(f"{dep_tool} final output missing: {output_path}")
        
        return len(missing_details) == 0, missing_details

    def run_tool(self, tool_name: str, tool_config: Dict[str, Any]):
        """Execute a single analysis tool."""
        self.log(f"\n{'=' * 3} Running {tool_name} {'=' * (60 - len(tool_name) - 12)}")
        
        with self.tool_logging_context(tool_name) as log_file:
            self.log(f"Logging run to: {log_file}")
            
            execution_mode = tool_config.get('execution_mode', self.get_setting('default_execution_mode'))
            
            if execution_mode == 'per_sample' and tool_config.get('sample_output_dir'):
                sample_log_pattern = self.substitute_variables(f"{tool_config['sample_output_dir']}/{{sample_id}}.{tool_name}.log", "*")
                self.log(f"Logging per-sample tool output to: {sample_log_pattern}")
            
            # Show execution mode and parallelism
            if execution_mode == 'per_sample':
                max_threads = tool_config.get('parallel', self.get_setting('default_parallel'))
                self.log(f"Run mode: {max_threads} samples in parallel")
            else:
                self.log(f"Run mode: batch")
            
            # Check dependencies
            deps_ok, missing_details = self.check_dependencies(tool_name)
            if not deps_ok:
                error_msg = f"{tool_name} cannot run due to missing dependencies:\n" + "\n".join(f"  - {detail}" for detail in missing_details)
                raise RuntimeError(error_msg)
            
            # Check if should skip due to existing outputs
            if self.check_and_skip_tool(tool_name, tool_config):
                return
            
            # Start execution
            start_time = time.time()
            self.log(f"{tool_name} Start: {self.get_timestamp()}")
            
            self.current_tool_modules = tool_config.get('modules', [])
            
            # Create output directory if needed
            if tool_config.get('output_dir') and not self.dry_run:
                Path(self.substitute_variables(tool_config['output_dir'])).mkdir(parents=True, exist_ok=True)
            
            # Execute based on mode
            memory_values, cpu_values = self._execute_tool_commands(tool_name, tool_config, execution_mode)
            
            # Mark tool as executed
            self.executed_tools.append(tool_name)
            
            # Complete and report
            self.log(f"✅ {tool_name} completed successfully")
            
            runtime = time.time() - start_time
            self._report_resource_usage(tool_name, execution_mode, runtime, memory_values, cpu_values)

    def _execute_tool_commands(self, tool_name: str, tool_config: Dict[str, Any], execution_mode: str) -> Tuple[List[float], List[float]]:
        """Execute tool commands and collect resource metrics."""
        memory_values = []
        cpu_values = []
        failed_samples = []
        
        def collect_metrics(cmd, context=None):
            mem, cpu, _, returncode = self.run_command(cmd, context=context)
            if self.monitor and mem > 0:
                memory_values.append(mem)
                cpu_values.append(cpu)
            return returncode
        
        # Pre-commands (don't collect metrics for resource tracking)
        for cmd in tool_config.get('pre_commands', []):
            returncode = self.run_command(cmd, context=f"pre-{tool_name} command")
            if returncode != 0:
                raise RuntimeError(f"Pre-command failed for {tool_name} with exit code {returncode}")
        
        if execution_mode == 'per_sample':
            # Parallel sample processing - only collect metrics from main command executions
            sample_ids = self.get_sample_ids()
            max_threads = tool_config.get('parallel', self.get_setting('default_parallel'))
            
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {executor.submit(self.process_sample, tool_name, tool_config, sid): sid for sid in sample_ids}
                for future in as_completed(futures):
                    result, mem, cpu, _ = future.result()
                    # Check if sample failed (but not allowed failures)
                    if "failed" in result and "allowed failure" not in result:
                        failed_samples.append(futures[future])
                    # Collect metrics from all non-skipped runs (including allowed failures that succeeded)
                    elif self.monitor and mem > 0 and "skipped" not in result:
                        memory_values.append(mem)
                        cpu_values.append(cpu)
            
            # Raise error if any samples failed (excluding allowed failures)
            if failed_samples:
                raise RuntimeError(f"{len(failed_samples)} sample(s) failed: {', '.join(failed_samples[:5])}")
        else:
            # Batch mode - collect metrics from main command only
            returncode = collect_metrics(self.substitute_variables(tool_config['command'], tool_config=tool_config), f"{tool_name} main command")
            if returncode != 0:
                raise RuntimeError(f"{tool_name} command failed with exit code {returncode}")
        
        # Post-commands (don't collect metrics for resource tracking)
        for cmd in tool_config.get('post_commands', []):
            self.run_command(cmd, context=f"post-{tool_name} command")
        
        return memory_values, cpu_values

    def create_versions_log(self):
        """Create or update versions log file with tool versions."""
        log_filename = f"{self.runid}.versions.tsv" if self.runid else "versions.tsv"
        versions_file = self.rundir / log_filename
        pipeline_path = Path(__file__).absolute()
        config_version = self.config.get('settings', {}).get('version', DEFAULTS['default_unknown_value'])
        
        # Write header and initial entries if new file or clean mode
        if self.clean or not versions_file.exists():
            timestamp = self.get_timestamp()
            with open(versions_file, 'w') as f:
                f.write("timestamp\ttool\tversion\tpath\tdatabase\n")
                f.write(f"{timestamp}\tbactopipe.py\t{VERSION}\t{pipeline_path}\t{DEFAULTS['default_database_value']}\n")
                f.write(f"{timestamp}\tbactopipe_config.yaml\t{config_version}\t{self.config_file}\t{DEFAULTS['default_database_value']}\n")

        # Store the filename for later reference
        self.versions_file = versions_file
        return versions_file

    def create_resource_log(self):
        """Create resource usage log file."""
        if not self.monitor:
            return None
            
        log_filename = f"{self.runid}.resource_usage.tsv" if self.runid else "resource_usage.tsv"
        resource_file = self.rundir / log_filename
        
        # Write header if new file or clean mode
        if self.clean or not resource_file.exists():
            header = "timestamp\ttool\texecution_mode\tn_ran\tmean_memory_gb\tmax_memory_gb\tmean_cpu_cores\tmax_cpu_cores\truntime_seconds\tparallel\tsuggested_threads\tmemory_t_recommended\tmemory_t_limit\tcpu_t_recommended\tcpu_t_limit\tbottleneck\tmem_total_percent\tcpu_total_percent\n"
            with open(resource_file, 'w') as f:
                f.write(header)
        
        self.resource_file = resource_file
        return resource_file

    def _report_resource_usage(self, tool_name: str, execution_mode: str, runtime: float, memory_values: List[float], cpu_values: List[float]):
        """Report and store resource usage data."""
        if self.monitor and memory_values:
            stats = {
                'tool': tool_name,
                'execution_mode': execution_mode,
                'n_ran': len(memory_values),
                'mean_memory_gb': sum(memory_values) / len(memory_values),
                'max_memory_gb': max(memory_values),
                'mean_cpu_cores': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max_cpu_cores': max(cpu_values) if cpu_values else 0,
                'runtime_seconds': runtime
            }
            
            # Combined single-line end message with all metrics
            self.log(f"{tool_name} End | Total Runtime: {self.format_runtime(runtime)} | Peak Memory: {stats['max_memory_gb']:.2f} GB | CPU: {stats['max_cpu_cores']:.1f} cores")
            self.resource_data.append(stats)
        else:
            # No resource data - just runtime
            self.log(f"{tool_name} End. Runtime: {self.format_runtime(runtime)}")

    def format_runtime(self, seconds: float) -> str:
        """Format runtime in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes}m {seconds % 60:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m {seconds % 60:.1f}s"

    def check_sample_outputs(self, sample_output_file: str, sample_ids: List[str]) -> tuple[List[str], List[str]]:
        """Check which samples have existing outputs."""
        missing_samples = []
        existing_samples = []
        
        for sample_id in sample_ids:
            sample_pattern = self.substitute_variables(sample_output_file, sample_id)
            
            # First try exact file path (no glob)
            if Path(sample_pattern).exists():
                existing_samples.append(sample_id)
            # If exact path doesn't exist and pattern contains wildcards, try glob
            elif '*' in sample_pattern or '?' in sample_pattern:
                matching_files = list(Path(sample_pattern).parent.glob(Path(sample_pattern).name))
                if matching_files:
                    existing_samples.append(sample_id)
                else:
                    missing_samples.append(sample_id)
            # No glob pattern and file doesn't exist
            else:
                missing_samples.append(sample_id)
        
        return missing_samples, existing_samples

    def _check_file_empty_or_header_only(self, file_path: Path) -> bool:
        """Check if file is empty or contains only header line."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            return len(lines) <= 1
        except Exception:
            return False  # If we can't read the file, assume it's ok

    def check_and_skip_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> bool:
        """Check if tool should be skipped based on existing outputs."""
        if self.force:
            return False
        
        # Simplified with helper method
        execution_mode = tool_config.get('execution_mode', self.get_setting('default_execution_mode'))
        
        output_file = tool_config.get('output_file')
        sample_output_file = tool_config.get('sample_output_file')
        
        # Check sample outputs
        if sample_output_file:
            sample_ids = self.get_sample_ids()
            missing_samples, existing_samples = self.check_sample_outputs(sample_output_file, sample_ids)
            
            if missing_samples:
                missing_count = len(missing_samples)
                sample_list = ', '.join(missing_samples[:5]) + ('...' if missing_count > 5 else '')
                self.log(f"⚠️  {tool_name}: missing outputs for {missing_count} samples: {sample_list}")
                return False
            
            if not output_file:
                # Create display pattern by replacing {sample_id} with *
                display_pattern = self.substitute_variables(sample_output_file.replace('{sample_id}', '*'))
                self.log(f"✅ {tool_name}: all {len(existing_samples)} sample outputs exist: {display_pattern}")
                self.log(f"✅ {tool_name} skipped.")
                return True
        
        # Check final output
        if output_file:
            output_path = Path(self.substitute_variables(output_file))
            if output_path.exists():
                # For per_sample mode with sample outputs already verified, check content
                if execution_mode == 'per_sample' and sample_output_file:
                    if self._check_file_empty_or_header_only(output_path):
                        self.log(f"⚠️  {tool_name}: final output exists but appears empty or header-only: {output_path}")
                        self.log(f"⚠️  {tool_name} skipped (but check output quality).")
                    else:
                        self.log(f"✅ {tool_name}: all sample outputs and final output exist")
                        self.log(f"✅ {tool_name} skipped.")
                else:
                    # Batch mode or no sample outputs
                    self.log(f"✅ {tool_name} skipped (output exists: {output_path})")
                return True
            elif sample_output_file:
                self.log(f"⚠️  {tool_name}: sample outputs exist but final output missing: {output_path}")
                return False
        
        return False

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting from config with fallback to DEFAULTS."""
        return self.config.get('settings', {}).get(key, default if default is not None else DEFAULTS.get(key))

    def calculate_resource_limits(self, max_memory_gb: float, max_cpu_cores: float) -> Optional[Dict[str, Any]]:
        """Calculate suggested parallelism based on system resources. Returns None if calculation not possible."""
        available_cores = self.system_resources.get('logical_cores', self.system_resources.get('cpu_cores', DEFAULTS['default_unknown_value']))
        available_memory = self.system_resources.get('memory_gb', DEFAULTS['default_unknown_value'])
        
        # Return None if we can't calculate (instead of dict of 'unknown' values)
        if (available_cores == DEFAULTS['default_unknown_value'] or 
            available_memory == DEFAULTS['default_unknown_value'] or 
            max_memory_gb <= 0 or 
            max_cpu_cores <= 0):
            return None
        
        utilization = DEFAULTS['resource_utilization_target']
        
        # Actual theoretical limits (100% system utilization)
        memory_limit = int(available_memory / max_memory_gb)
        cpu_limit = int(available_cores / max_cpu_cores)
        
        # Recommended limits (80% system utilization)
        memory_recommended = int(available_memory * utilization / max_memory_gb)
        cpu_recommended = int(available_cores * utilization / max_cpu_cores)
        
        suggested = min(memory_recommended, cpu_recommended)
        bottleneck = 'memory' if memory_recommended < cpu_recommended else 'cpu'
        
        return {
            'suggested': suggested,
            'memory_limit': memory_limit,
            'cpu_limit': cpu_limit,
            'memory_recommended': memory_recommended,
            'cpu_recommended': cpu_recommended,
            'bottleneck': bottleneck,
            'mem_total_percent': round((suggested * max_memory_gb / available_memory * 100), 1),
            'cpu_total_percent': round((suggested * max_cpu_cores / available_cores * 100), 1)
        }

    def get_system_resources(self) -> Dict[str, Any]:
        """Query system resources and return available CPU cores and memory."""
        resources = {'cpu_cores': 'unknown', 'logical_cores': 'unknown', 'memory_gb': 'unknown', 
                     'memory_used_gb': 'unknown', 'memory_free_gb': 'unknown', 
                     'cpu_load_5min': 'unknown'}

        try:
            # Get logical cores from nproc
            result = subprocess.run(['nproc'], capture_output=True, text=True)
            if result.returncode == 0:
                resources['logical_cores'] = int(result.stdout.strip())
                
                # Try to get physical cores from lscpu
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                if result.returncode == 0:
                    cores_per_socket = sockets = None
                    for line in result.stdout.split('\n'):
                        if line.startswith('Core(s) per socket:'):
                            cores_per_socket = int(line.split(':')[1].strip())
                        elif line.startswith('Socket(s):'):
                            sockets = int(line.split(':')[1].strip())
                    
                    resources['cpu_cores'] = cores_per_socket * sockets if cores_per_socket and sockets else resources['logical_cores']
                else:
                    resources['cpu_cores'] = resources['logical_cores']
            
            # Get memory info (total, used, free)
            result = subprocess.run(['free', '-b'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Mem:'):
                        parts = line.split()
                        resources['memory_gb'] = round(int(parts[1]) / (1024**3), 1)
                        resources['memory_used_gb'] = round(int(parts[2]) / (1024**3), 1)
                        resources['memory_free_gb'] = round(int(parts[3]) / (1024**3), 1)
                        break
            
            # Get 5-minute CPU load average
            result = subprocess.run(['uptime'], capture_output=True, text=True)
            if result.returncode == 0:
                # uptime output: "... load average: 1.23, 4.56, 7.89"
                if 'load average:' in result.stdout:
                    load_str = result.stdout.split('load average:')[1].strip()
                    loads = [float(x.strip()) for x in load_str.split(',')]
                    if len(loads) >= 2:
                        resources['cpu_load_5min'] = loads[1]  # Second value is 5-min average
        except Exception:
            pass  # Keep unknown values
        
        return resources

    def parse_time_output(self, stderr: str) -> Tuple[float, float, float]:
        """Parse /usr/bin/time -v output and return (peak_memory_gb, cpu_cores, user_time)."""
        metrics = {'peak_memory_gb': 0.0, 'user_time': 0.0, 'system_time': 0.0, 'wall_time': 0.0}
        
        for line in stderr.split('\n'):
            if 'Maximum resident set size (kbytes):' in line:
                metrics['peak_memory_gb'] = float(line.split(':')[1].strip()) / 1024 / 1024
            elif 'User time (seconds):' in line:
                try:
                    metrics['user_time'] = float(line.split(':')[1].strip())
                except ValueError:
                    pass
            elif 'System time (seconds):' in line:
                try:
                    metrics['system_time'] = float(line.split(':')[1].strip())
                except ValueError:
                    pass
            elif 'Elapsed (wall clock) time' in line:
                # /usr/bin/time -v outputs in h:mm:ss.ss or m:ss.ss format
                time_str = line.split('):', 1)[1].strip()
                parts = [int(float(p)) if '.' not in p else float(p) for p in time_str.split(':')]
                metrics['wall_time'] = sum(part * (60 ** (len(parts) - 1 - i)) for i, part in enumerate(parts))
        
        # Calculate average CPU cores used (total CPU time / wall time)
        cpu_cores = (metrics['user_time'] + metrics['system_time']) / metrics['wall_time'] if metrics['wall_time'] > 0 else 0.0
        
        return metrics['peak_memory_gb'], cpu_cores, metrics['user_time']

    def setup_run_directory(self):
        """Execute setup commands."""
        if 'setup_commands' not in self.config:
            return
        
        self.log(f"\n{'=' * 3} Running setup {'=' * 44}")
        
        for command in self.config['setup_commands']:
            full_command = self.substitute_variables(command)
            self.log(f"Setup: {full_command}")
            
            if not self.dry_run:
                result = subprocess.run(
                    full_command, shell=True, executable='/bin/bash', capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Setup command failed: {full_command}")
        
        self.log("✅ Setup completed successfully")

    def cleanup_run_directory(self):
        """Execute cleanup commands at the end of pipeline execution."""
        if 'cleanup_commands' not in self.config:
            return

        self.log(f"\n{'=' * 3} Running cleanup {'=' * 42}")
        # Log details to pipeline log only (not console, not tool logs)
        for command in self.config['cleanup_commands']:
            full_command = self.substitute_variables(command)
            if self.pipeline_log_file:
                self.write_to_file(f"Cleanup: {full_command}", self.pipeline_log_file)
            
            if not self.dry_run:
                result = subprocess.run(
                    full_command, shell=True, executable='/bin/bash', capture_output=True, text=True
                )
                # Don't fail the pipeline if cleanup fails, just log it
                if result.returncode != 0:
                    if self.pipeline_log_file:
                        self.write_to_file(f"Warning: Cleanup command failed: {full_command}", self.pipeline_log_file)
        
        self.log("✅ Cleanup completed successfully")

    def run_pipeline(self, tools: Optional[List[str]] = None):
        """Execute the pipeline."""
       
        # Create output directory first if it doesn't exist
        if not self.dry_run:
            self.rundir.mkdir(parents=True, exist_ok=True)
        
        self.setup_pipeline_logging()
        self.print_startup_info()
        
        # Determine what tools to run
        tools_to_run = tools if tools else list(self.config['tools'].keys())
        
        self.create_versions_log()
        self.create_resource_log()
        
        self.log(f"\nTools to run: {tools_to_run}")
        
        # Run setup unless in tools mode
        if not tools:
            self.setup_run_directory()
        
        # Run tools
        for tool_name in tools_to_run:
            try:
                if tool_name in self.config['tools']:
                    self.run_tool(tool_name, self.config['tools'][tool_name])
                else:
                    self.log(f"⚠️  Unknown tool: {tool_name}")
            except Exception as e:
                self.log(f"❌ {tool_name} failed: {e}")
                sys.exit(1)

        # Run cleanup unless in tools mode
        if not tools:
            self.cleanup_run_directory()
        
        self.current_log_file = None
        
        # Write resource data
        self.write_resource_data()
        
        # Log versions only for tools that actually executed
        if self.executed_tools:
            self.log(f"\n{'=' * 3} Logging tool versions {'=' * 37}")
            for tool_name in self.executed_tools:
                self.log_tool_version(tool_name, self.config['tools'][tool_name])
            
            self.log(f"\nTool versions appended to: {self.versions_file}")
        
        if self.monitor:
            self.log(f"Resource usage appended to: {self.resource_file}")
        
        # Calculate total pipeline runtime
        total_runtime = time.time() - self.pipeline_start_time
        
        # Summary
        tools_summary = f"specified tools ({', '.join(tools_to_run)})" if tools else "all tools"
        self.log("\n" + "=" * 50)
        self.log(f"Pipeline completed for {self.runid} - {tools_summary}")
        self.log(f"Total runtime: {self.format_runtime(total_runtime)}")
        self.log(f"Timestamp: {self.get_timestamp()}")
        self.log("=" * 50)

    def write_resource_data(self):
        """Write accumulated resource data to file."""
        if not self.monitor or not self.resource_data:
            return
        
        timestamp = self.get_timestamp()
        
        with open(self.resource_file, 'a') as f:
            for data in self.resource_data:
                # Get current parallel setting
                tool_config = self.config['tools'].get(data['tool'], {})
                parallel = tool_config.get('parallel', self.get_setting('default_parallel'))
                
                # Calculate resource limits and suggestions
                limits = self.calculate_resource_limits(data['max_memory_gb'], data['max_cpu_cores'])
                
                # Write timestamp and basic stats
                f.write(f"{timestamp}\t{data['tool']}\t{data['execution_mode']}\t{data['n_ran']}\t")
                f.write(f"{data['mean_memory_gb']:.3f}\t{data['max_memory_gb']:.3f}\t")
                f.write(f"{data['mean_cpu_cores']:.1f}\t{data['max_cpu_cores']:.1f}\t")
                f.write(f"{data['runtime_seconds']:.1f}\t{parallel}\t")
                
                # Write resource calculations or 'unknown' placeholders
                if limits is None:
                    # Write 'unknown' for all calculated fields
                    f.write(f"{DEFAULTS['default_unknown_value']}\t" * 8 + "\n")
                else:
                    f.write(f"{limits['suggested']}\t")
                    f.write(f"{limits['memory_recommended']}\t{limits['memory_limit']}\t")
                    f.write(f"{limits['cpu_recommended']}\t{limits['cpu_limit']}\t")
                    f.write(f"{limits['bottleneck']}\t")
                    f.write(f"{limits['mem_total_percent']}\t{limits['cpu_total_percent']}\n")

    def get_tool_version_info(self, tool_name: str, tool_config: Dict[str, Any]) -> tuple[str, str, str]:
        """Extract version information for a tool."""
        version = tool_path = DEFAULTS['default_unknown_value']
        database = DEFAULTS['default_database_value']
        
        try:
            modules = tool_config.get('modules', [])
            
            # Helper to add module setup to commands
            def add_modules(cmd: str) -> str:
                if modules:
                    module_setup = " && ".join(["module purge"] + [f"module load {m}" for m in modules])
                    return f". /etc/profile.d/modules.sh; {module_setup} && {cmd}"
                return cmd
            
            # Get version command
            default_cmd = DEFAULTS['default_version_cmd_pattern'].replace('{tool_name}', tool_name)
            version_cmd = self.substitute_variables(tool_config.get('version_cmd', default_cmd), tool_config=tool_config)
            version_cmd = add_modules(version_cmd)
            
            # Run version command
            result = subprocess.run(version_cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)
            if result.returncode == 0:
                output = result.stdout.strip() if result.stdout else result.stderr.strip()
                if output:
                    version = output.split('\n')[0]
            
            # Get tool path - use tool_path from config if available, then tool_path_cmd, otherwise which
            if 'tool_path' in tool_config:
                tool_path = self.substitute_variables(tool_config['tool_path'], tool_config=tool_config)
            elif 'tool_path_cmd' in tool_config:
                path_cmd = self.substitute_variables(tool_config['tool_path_cmd'], tool_config=tool_config)
                path_cmd = add_modules(path_cmd)
                
                result = subprocess.run(path_cmd, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=DEFAULTS['subprocess_timeout'])
                if result.returncode == 0:
                    tool_path = result.stdout.strip()
            else:
                path_cmd = add_modules(f"which {tool_name}")
                
                result = subprocess.run(path_cmd, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=DEFAULTS['subprocess_timeout'])
                if result.returncode == 0:
                    tool_path = result.stdout.strip()
            
            # Get database information
            if 'db_path' in tool_config:
                database = self.substitute_variables(tool_config['db_path'], tool_config=tool_config)
            elif 'db_cmd' in tool_config:
                db_cmd = self.substitute_variables(tool_config['db_cmd'], tool_config=tool_config)
                db_cmd = add_modules(db_cmd)
                
                result = subprocess.run(db_cmd, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=DEFAULTS['subprocess_timeout'])
                if result.returncode == 0 and result.stdout.strip():
                    database = result.stdout.strip().split('\n')[0]
                
        except Exception:
            pass  # Keep defaults
        
        return version, tool_path, database

    def log_tool_version(self, tool_name: str, tool_config: Dict[str, Any]):
        """Log version information for a tool to the versions file."""
        
        timestamp = self.get_timestamp()
        
        # Log main tool
        version, tool_path, database = self.get_tool_version_info(tool_name, tool_config)
        with open(self.versions_file, 'a') as f:
            f.write(f"{timestamp}\t{tool_name}\t{version}\t{tool_path}\t{database}\n")
        
        # Log sub-tools if defined
        for sub_tool in tool_config.get('sub_tools', []):
            sub_name = f"{tool_name}:{sub_tool['name']}"
            sub_config = {
                'modules': tool_config.get('modules', []),
                **{k: v for k, v in sub_tool.items() if k != 'name' and v is not None} 
            }
            
            version, tool_path, database = self.get_tool_version_info(sub_tool['name'], sub_config)
            with open(self.versions_file, 'a') as f:
                f.write(f"{timestamp}\t{sub_name}\t{version}\t{tool_path}\t{database}\n")
    
    @staticmethod
    def print_tool_versions(config_file: str):
        """Print tool versions for all tools in the configuration."""
        config_path = Path(config_file) if Path(config_file).is_absolute() else Path(__file__).parent / config_file
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("timestamp\ttool\tversion\tpath\tdatabase")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{timestamp}\tbactopipe.py\t{VERSION}\t{Path(__file__).absolute()}\tnone")
        config_version = config.get('settings', {}).get('version', 'unknown')
        print(f"{timestamp}\tbactopipe_config.yaml\t{config_version}\t{config_path}\tnone")
        
        # Create minimal runner for version checking
        runner = PipelineRunner(str(config_path), output_dir="version_check_temp", dry_run=True)
        
        for tool_name, tool_config in config['tools'].items():
            version, path, db = runner.get_tool_version_info(tool_name, tool_config)
            print(f"{timestamp}\t{tool_name}\t{version}\t{path}\t{db}")
            
            # Check for sub-tools
            for sub_tool in tool_config.get('sub_tools', []):
                sub_name = f"{tool_name}:{sub_tool['name']}"
                sub_config = {
                    'modules': tool_config.get('modules', []),
                    **{k: v for k, v in sub_tool.items() if k != 'name' and v is not None} 
                }
                
                version, path, db = runner.get_tool_version_info(sub_tool['name'], sub_config)
                print(f"{timestamp}\t{sub_name}\t{version}\t{path}\t{db}")

def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(description='Genomic analysis pipeline')
    parser.add_argument('-r', '--runid', help='Run ID')
    parser.add_argument('-i', '--input-file', dest='input_file', help='Input samples TSV file')
    parser.add_argument('--input-dir', dest='input_dir', help='Input directory (for setup)')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Output directory')
    parser.add_argument('-c', '--config', default='bactopipe_config.yaml', help='Config file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (only show commands don\'t run them)')
    parser.add_argument('--force', action='store_true', help='Force overwrite of existing output')
    parser.add_argument('--skip', type=str, help='Comma-separated list of sample IDs to allow to fail (e.g., "NEG,Lambda,sample3")')
    parser.add_argument('--clean', action='store_true', help='Overwrite log files instead of appending')
    parser.add_argument('--tools', nargs='+', help='Tools to run')
    parser.add_argument('--tool_versions', action='store_true', help='Show tool versions only')
    parser.add_argument('--monitor', action='store_true', default=True, help='Monitor resource usage (default: enabled)')
    parser.add_argument('--verbose', action='store_true', help='Print commands to console')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {VERSION}', help='Show program version')
    
    args = parser.parse_args()
    
    # Handle tool versions only mode
    if args.tool_versions:
        PipelineRunner.print_tool_versions(args.config)
        return
    
    # Validate argument combinations (only for non-tool_versions mode)
    if not args.output_dir:
        sys.exit("ERROR: --output-dir (-o) is required")
    if not args.input_dir and not args.input_file:
        sys.exit("ERROR: Either --input-dir or --input-file must be provided")
    
    config_path = Path(__file__).parent / args.config if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.exists():
        sys.exit(f"ERROR: Config file not found: {config_path}")
    
    run = PipelineRunner(
        config_file=str(config_path),
        input_file=args.input_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        runid=args.runid,  # Pass the runid argument
        dry_run=args.dry_run,
        force=args.force,
        skip_samples=args.skip,
        clean=args.clean,
        monitor=args.monitor,
        verbose=args.verbose
    )
    
    run.run_pipeline(args.tools)

if __name__ == '__main__':
    main()
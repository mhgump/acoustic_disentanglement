import argparse
import subprocess
import os
from os.path import dirname

from speech_representations.features import list_features_by_tags


def parse_args(args=None):
    """ This script runs the component scripts necessary to create a split dataset.

    These arguments control how these scripts are run. For the full set of arguments controlling the dataset file created,
    see speech_representations/data/datasource_writer.py. source_directory, target_directory, dataset_name must
    always be included.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source_directory', type=str, help='Directory holding source data')
    parser.add_argument('target_directory', type=str, help='Directory where datafiles should be written')
    parser.add_argument('dataset_name', type=str, help='Name of dataset, must match parser implementation')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of split jobs to start in slurm')
    parser.add_argument('--project_directory', type=str, required=False,
        help='Directory within this project containing slurm files, by default uses __file__ (which might not work)')
    parser.add_argument('--use_slurm', action='store_true', help='Whether to use slurm to complete jobs')
    parser.add_argument('--slurm_logs_directory', type=str, required=False, help='Directory to write slurm logs in')
    parser.add_argument('--dry_run', action='store_true', help='Don\'t run anything just log what would be run.')
    parser.add_argument('--verbose', action='store_true', help='Print out script output for debugging.')
    if args:
        return parser.parse_known_args(args)
    else:
        return parser.parse_known_args()


def append_sbatch_command(command, logdir, name, dependencies=None):
    prefix = ' '.join(['sbatch',
                       '--parsable',
                       '--output={}/{}_%j.out'.format(logdir, name),
                       '--error={}/{}_%j.err'.format(logdir, name),
                       '--job-name={}'.format(name),
                       '--kill-on-invalid-dep=yes',])
    if dependencies != None:
        dependency_string = '--dependency=afterok:{}'.format(':'.join(map(str, dependencies)))
        prefix = ' '.join([prefix, dependency_string])
    return ' '.join([prefix, command])


def run_command(command, dry_run, verbose=False):
    if not dry_run:
        command_result = subprocess.run(command, stdout=subprocess.PIPE)
        command_output = command_result.stdout.decode('utf-8').strip()
    if verbose:
        print('Running: ')
        print(command)
        print()
        if not dry_run:
           print('Output: ')
           print(command_output)
           print()
    if (not dry_run) and command_output.isdigit():
        return int(command_output)
    else:
        return 0


def run(args, unknown_args):
    if args.project_directory is None and '__file__' not in globals():
        raise RuntimeException('__file__ is not working correctly for this module, you must pass the \
            project_directory explicitly.')
    project_directory = args.project_directory if args.project_directory is not None \
        else dirname(dirname(dirname(os.path.join('.', __file__))))    

    initialize_script = os.path.join(project_directory, 'slurm_templates/initialize_dataset.sh')
    split_script = os.path.join(project_directory, 'slurm_templates/process_dataset_split.sh')
    finalize_script = os.path.join(project_directory, 'slurm_templates/finalize_dataset.sh')
    entrypoint_script = os.path.join(project_directory, 'speech_representations/data/datasource_writer.py')

    args_prefix = [args.source_directory, args.target_directory, args.dataset_name]
    intialize_command = " ".join([initialize_script, entrypoint_script, *args_prefix, 'setup_job', *unknown_args])
    split_command = " ".join([split_script, entrypoint_script, *args_prefix, 'start_split_process', *unknown_args, \
        '--num_processes={}'.format(args.num_processes), '--process_index=__PROCESS_INDEX__'])
    finalize_command = " ".join([finalize_script, entrypoint_script, *args_prefix, 'finalize_job', *unknown_args])

    if args.use_slurm:
        intialize_command = append_sbatch_command(intialize_command, args.slurm_logs_directory, 'initialize_dataset')
    initialize_id = run_command(intialize_command, args.dry_run, verbose=args.verbose)
    if args.use_slurm:
        split_command = append_sbatch_command(split_command, args.slurm_logs_directory, 'process_dataset_split', [initialize_id])
    split_ids = []
    for index in range(args.num_processes):
        this_split_command = split_command.replace('__PROCESS_INDEX__', str(index))
        split_id = run_command(this_split_command, args.dry_run, verbose=args.verbose)
        split_ids += [split_id]
    if args.use_slurm:
        finalize_command = append_sbatch_command(finalize_command, args.slurm_logs_directory, 'finalize_dataset', split_ids)
    finalize_id = run_command(finalize_command, args.dry_run, verbose=args.verbose)


if __name__ == '__main__':
    args, unknown_args = parse_args()
    run(args, unknown_args)

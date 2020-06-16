import threading
import subprocess
from collections import defaultdict
from collections import ChainMap
from transformers import (
    HfArgumentParser
)
import copy
from configuration_classes import (
    parameters_to_command,
    parameters_to_string,
    ModelArguments, 
    DataEvaluationArguments, 
    BeamsearchArguments, 
    RankingArguments,
    BeamsearchManagerArguments
)

import string
import torch.cuda
import shutil
import itertools
import json 
import logging 
import os
import sys

def main():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, BeamsearchArguments, RankingArguments, BeamsearchManagerArguments))
    model_args, data_args, beam_args, ranking_args, manager_args = parser.parse_args_into_dataclasses()

    if os.path.isfile(data_args.dict_file):
        sys.exit()

    print(parameters_to_string(model_args, data_args, beam_args, ranking_args))

    # Calculate required workers
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    total_num_workers = int(total_memory_bytes) // int(manager_args.expected_worker_load)
    total_num_workers = int(total_num_workers)
    cmd = parameters_to_command('beamsearch.py', model_args, data_args, beam_args, ranking_args)

    output_parameters = [
        '--dict_file',
        '--expansions_file',
        '--logfile'
    ]

    queue = []
    for worker_index in range(total_num_workers):
        worker_cmd = copy.copy(cmd)
        for idx, parameter in enumerate(worker_cmd):
            if parameter not in ['python', 'beamsearch.py']:
                flag = parameter.split('=')[0]
                old_value = parameter.split('=')[1]
            else:
                continue
            # Change n_chunks, index parameters
            if parameter.startswith('--n_chunks'):
                new_parameter = '{0}={1}'.format(flag, total_num_workers)
                worker_cmd[idx] = new_parameter

            elif parameter.startswith('--index'):
                new_parameter = '{0}={1}'.format(flag, worker_index)
                worker_cmd[idx] = new_parameter

            # Change output parameters
            elif any(parameter.startswith(x) for x in output_parameters):
                def compute_parameter(parameter):
                    flag_and_filename, ending = parameter.split('.')
                    new_flag_and_filename = flag_and_filename + '_{0}'.format(worker_index)
                    new_parameter = '{0}.{1}'.format(
                        new_flag_and_filename,
                        ending
                    )
                    return new_parameter
                worker_cmd[idx] = compute_parameter(parameter)
        queue.append(worker_cmd)

    thread_queue = [ threading.Thread(target=subprocess.call, args=(cmd, )) \
        for cmd in queue]
    for thread in thread_queue:
        thread.start()
    for thread in thread_queue:
        thread.join()

    #Merge outputs

    output_parameters = [
        '--dict_file',
        '--expansions_file',
        '--logfile'
    ]

    merge_dict = defaultdict(list)
    for cmd in queue:
        for parameter in cmd:
            for flag in output_parameters:
                if parameter.startswith(flag):
                    merge_dict[flag].append(parameter.split(flag)[1].lstrip('='))

    for value in merge_dict.values():
        try:
            first = value[0]
            target_destination, file_extension = first.split('.')
            target_destination = target_destination.strip('_' + string.digits) + '.' + file_extension
            if first.endswith('.json'):
                content = []
                for item in value:
                    with open(item, 'r') as f:
                        content.append(json.load(f))

                if isinstance(content[0], list):
                    output = list(itertools.chain(*content))
                elif isinstance(content[0], dict):
                    output = dict(ChainMap(*content))
                else:
                    raise NotImplementedError
                
                with open(target_destination, 'w+') as f:
                    json.dump(output, f)

            else:
                with open(target_destination, 'a+') as target_f:
                    for filename in value:
                        with open(filename,'r') as file_f:
                            filename_content = file_f.read()
                            print(filename_content, file=target_f)
        except:
            continue

if __name__ == '__main__':
    main()
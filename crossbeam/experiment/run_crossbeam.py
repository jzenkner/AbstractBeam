# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import argparse
import functools
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import torch
import random
from crossbeam.dsl import task as task_module
from crossbeam.data.deepcoder import deepcoder_tasks
from ml_collections import config_flags
from crossbeam.datasets import data_gen
from crossbeam.common import configs_all
from crossbeam.dsl import domains
from crossbeam.experiment.exp_common import set_global_seed
from crossbeam.experiment.train_eval import main_train_eval
from crossbeam.model.joint_model import JointModel, IntJointModel
from crossbeam.model.logic_model import LogicModel
from crossbeam.model.deepcoder_model import DeepCoderModel
from crossbeam.model.util import CharacterTable
from crossbeam.experiment.deepcoder.configs.baseline_eval import get_config

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string='Path to the Training configuration.')

flags.DEFINE_string('model_type', 'char', 'int/char/logic')
flags.DEFINE_bool('stochastic_beam', False, 'do stochastic beam search during test')
flags.DEFINE_bool('random_beam', False, 'replace beam search with random choices?')
flags.DEFINE_float('restarts_timeout', None, 'Timeout per random restart')
flags.DEFINE_float('temperature', 1.0, 'Temperature for sampling or UR evaluation')
flags.DEFINE_bool('synthetic_test_tasks', False, 'Use synthetic or handwritten test tasks.')


def init_model(args, domain, model_type, ckpt_inventions=[]):
  """Initializes the model."""
  if model_type.startswith('char'):
    input_table = CharacterTable(domain.input_charset,
                                 max_len=domain.input_max_len)
    output_table = CharacterTable(domain.output_charset,
                                  max_len=domain.output_max_len)
    value_table = CharacterTable(domain.value_charset,
                                 max_len=domain.value_max_len)
    return JointModel(args, input_table, output_table, value_table,
                      domain.operations)
  elif model_type.startswith('int'):
    return IntJointModel(args,
                         input_range=(0, 10),
                         output_range=(-800, 800),
                         value_range=(-800, 800),
                         operations=domain.operations)
  elif model_type.startswith('logic'):
    return LogicModel(args, operations=domain.operations)
  elif model_type == 'deepcoder':
    return DeepCoderModel(args, operations=domain.operations, ckpt_inventions = ckpt_inventions)
  else:
    raise ValueError('unknown model type %s' % model_type)


def get_eval_tasks(config):
  if config.do_test:
    if config.domain == 'deepcoder':
      return (deepcoder_tasks.SYNTHETIC_TASKS if config.synthetic_test_tasks
              else deepcoder_tasks.HANDWRITTEN_TASKS)
    eval_prefix = 'test-tasks'
  else:
    eval_prefix = 'vaxid-'
  eval_files = os.listdir(config.data_folder)
  eval_files = [fname for fname in eval_files if fname.startswith(eval_prefix)]
  eval_tasks = []
  for fname in sorted(eval_files):
    with open(os.path.join(config.data_folder, fname), 'rb') as f:
      eval_tasks += pickle.load(f)
    # Shuffle the evaluation tasks so that when we take the first `num_valid`
    # tasks, they come from different data-generation searches.
    random.shuffle(eval_tasks)
  return eval_tasks


def get_dreamcoder_eval_tasks(testing = False):
    Task = task_module.Task
    with open("/work/ldierkes/repos/ma-lukas-dierkes/ec/data/list_tasks+bootstrap.json", 'r') as file:
        data = json.load(file) 
    data = [task for task in data if task["type"]["input"] == task["type"]["output"] == "list-of-int"] # 
    names = [task["name"] for task in data]
    inputs = [{"x1": [task["examples"][i]["i"] for i in range(len(task["examples"]))]} for task in data]
    outputs = [[task["examples"][i]["o"] for i in range(len(task["examples"]))] for task in data]
    eval_tasks = [Task(name = name, inputs_dict=inputs[i], outputs = outputs[i], solution=None) for i, name in enumerate(names) if inputs[i] != outputs[i]]
    random.shuffle(eval_tasks)
    if testing:
      # take half 
      eval_tasks = eval_tasks[:len(eval_tasks)//2]
    else:
      # take other half
      eval_tasks = eval_tasks[len(eval_tasks)//2:]
    return eval_tasks


def main(argv):
  del argv
  if FLAGS.config is None: #: TODO: Config einbauen
    config = FLAGS
    proc_args = argparse.Namespace(**FLAGS.flag_values_dict())
  else:
    config = configs_all.get_config()
    config.update(get_config()) # FLAGS.config
    proc_args = config
    config.data_folder = os.path.join(config.data_root, config.data_name)
  logging.info(proc_args)
  set_global_seed(config.seed)
  domain = domains.get_domain(config.domain)
  
  if config.do_test:
    ckpt_file = os.path.join(config.save_dir, 'model-best-valid.ckpt')
  else:
    ckpt_file = os.path.join(config.save_dir, 'model-latest.ckpt')

  if config.load_model:
    ckpt_file = os.path.join(config.save_dir, config.load_model)
  if os.path.exists(ckpt_file):
    print('loading model from', ckpt_file)
    ckpt = torch.load(ckpt_file)
    print('model loaded at step %d' % ckpt['step'])
  else:
    ckpt = None

  domain = ckpt["domain"] if ckpt is not None else domain

  if ckpt is not None:
    model = init_model(config, domain, config.model_type, ckpt["inventions"])
  else:
    model = init_model(config, domain, config.model_type, ckpt_inventions=[])
  
  if config.dreamcoder:
    if config.do_test:
      # Load tasks from pickle
      with open("/work/ldierkes/repos/new/LambdaBeam/crossbeam/data/dreamcoder_test_tasks.pkl", 'rb') as file:
        original_tasks = pickle.load(file)
    else:
      with open("/work/ldierkes/repos/new/LambdaBeam/crossbeam/data/dreamcoder_train_tasks.pkl", 'rb') as file:
        original_tasks = pickle.load(file)
  else:
    original_tasks = deepcoder_tasks.HANDWRITTEN_TASKS
  print(len(original_tasks))
  print("config.save_dir", config.save_dir)
  print(f'Starting training, will save model dumps to {config.save_dir}')
  main_train_eval(proc_args, model,
                  trace_gen=data_gen.trace_gen,
                  checkpoint=ckpt, original_tasks = original_tasks)


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  app.run(main)


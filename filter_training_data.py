import collections
import glob
import multiprocessing
import os
import pickle as cp

from crossbeam.data.deepcoder import deepcoder_tasks
from crossbeam.dsl import deepcoder_utils
import random
import json
from crossbeam.dsl import task as task_module

NUM_PROCESSES = 14
DATA_DIR = '/work/ldierkes/repos/ma-lukas-dierkes/ec/LambdaBeam/t-3600-maxne-5-maxni-1-skip-0.0-lambdaskip-0.0-lambdafrac-0.8-shuffleops-False-now'
NEW_DATA_DIR = '/work/ldierkes/repos/ma-lukas-dierkes/ec/LambdaBeam/t-3600-maxne-5-maxni-1-skip-0.0-lambdaskip-0.0-lambdafrac-0.8-shuffleops-False-now/filtered_training/'


def task_type_signature(task):
  return (tuple(type(v[0]) for v in task.inputs_dict.values()),
          type(task.outputs[0]))

def get_dreamcoder_eval_tasks():
    Task = task_module.Task
    with open("/work/ldierkes/repos/ma-lukas-dierkes/ec/data/list_tasks+bootstrap.json", 'r') as file:
        data = json.load(file) 
    data = [task for task in data if task["type"]["input"] == task["type"]["output"] == "list-of-int"]
    names = [task["name"] for task in data]
    inputs = [{"x1": [task["examples"][i]["i"] for i in range(len(task["examples"]))]} for task in data]
    outputs = [[task["examples"][i]["o"] for i in range(len(task["examples"]))] for task in data]
    eval_tasks = [Task(name = name, inputs_dict=inputs[i], outputs = outputs[i], solution=None) for i, name in enumerate(names)]
    random.shuffle(eval_tasks)
    print("Len eval task when loaded")
    print(len(eval_tasks))
    return eval_tasks

TEST_TASKS = get_dreamcoder_eval_tasks()
TEST_TASKS_BY_TYPE_SIGNATURE = collections.defaultdict(list)
for task in TEST_TASKS:
  type_signature = task_type_signature(task)
  # Rename variables to the convention used in the training data.
  inputs_dict = {
      f'x{i + 1}': values
      for i, values in enumerate(task.inputs_dict.values())
  }
  outputs = task.outputs
  TEST_TASKS_BY_TYPE_SIGNATURE[type_signature].append((inputs_dict, outputs))


def process_shard(shard):
  shard_name = shard.split('/')[-1]
  with open(shard, 'rb') as f:
    tasks = cp.load(f)

  filtered_tasks = []
  for task in tasks:
    skip_task = False
    solution = task.solution.expression()
    type_signature = task_type_signature(task)
    for inputs_dict, outputs in TEST_TASKS_BY_TYPE_SIGNATURE[type_signature]:
      try:
        actual_outputs = deepcoder_utils.run_program(solution, inputs_dict)
        if actual_outputs == outputs:
          skip_task = True
          break
      except Exception:
        pass
    if not skip_task:
      filtered_tasks.append(task)

  with open(os.path.join(NEW_DATA_DIR, shard_name), 'wb') as f:
    cp.dump(filtered_tasks, f, cp.HIGHEST_PROTOCOL)

  num_excluded = len(tasks) - len(filtered_tasks)
  num_tasks = len(filtered_tasks)
  print(f'Finished shard {shard_name}, excluded {num_excluded} tasks, '
        f'{num_tasks} remain.')
  return num_tasks


def process_all_shards():
  shards = sorted(glob.glob(os.path.join(DATA_DIR, 'train-weight*.pkl')))
  with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
    num_tasks_per_shard = pool.map(process_shard, shards)
    print(f'The filtered data has {sum(num_tasks_per_shard)} tasks in total.')


if __name__ == '__main__':
  process_all_shards()

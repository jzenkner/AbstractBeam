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

"""Flags for data generation."""

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('data_gen_seed', 1, 'Seed for data generation')
flags.DEFINE_integer('shard_start_index', 0,
                     'starting index of shards for current job')
flags.DEFINE_enum('domain', 'tuple',
                  ['tuple', 'arithmetic', 'bustle', 'logic', 'deepcoder'],
                  'task domain')
flags.DEFINE_string('data_save_dir', None, 'Where to save data')
flags.DEFINE_string('split', 'train', 'the split for dataset generation')
flags.DEFINE_integer('num_datagen_proc', 1, '# processes for data gen')
flags.DEFINE_integer('shard_size', 100000, '# tasks per file')
flags.DEFINE_integer('num_tasks_per_weight', 1000, '# tasks for each weight')
flags.DEFINE_integer('num_searches', 100, '# searches to perform')
flags.DEFINE_float('data_gen_timeout', 60, 'timeout per search in seconds')
flags.DEFINE_integer('min_num_examples', 2, '')
flags.DEFINE_integer('max_num_examples', 4, '')
flags.DEFINE_integer('min_num_inputs', 1, '')
flags.DEFINE_integer('max_num_inputs', 3, '')
flags.DEFINE_integer('min_task_weight', 3, '')
flags.DEFINE_integer('max_task_weight', 10, '')
flags.DEFINE_float('skip_probability', 0,
                   'Probability of skipping a program during bottom-up '
                   'enumeration to enable seeing larger programs')
flags.DEFINE_float('lambda_skip_probability', 0,
                   'Probability of skipping a lambda program during bottom-up '
                   'enumeration to enable seeing larger programs')
flags.DEFINE_float('lambda_fraction', None,
                   'The proportion of values in the dataset that use lambdas.')
flags.DEFINE_boolean('shuffle_ops', False,
                     'Whether to shuffle the order of ops considered during '
                     'bottom-up search for data generation. If False, we might '
                     'throw away values of the largest weight to avoid biasing '
                     'toward values using ops considered first. If True, we do '
                     'not throw away anything, but the distribution of tasks '
                     'might change due to the different ordering of ops.')
flags.DEFINE_boolean('verbose', False, 'whether to print generated tasks')

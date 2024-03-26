"""Evaluate property signature quality with heuristics."""

import collections
import cProfile
import timeit

from absl import app

from crossbeam.algorithm import baseline_enumeration
from crossbeam.dsl import domains
from crossbeam.dsl import task as task_module
from crossbeam.dsl import value as value_module
from crossbeam.property_signatures import property_signatures

VERBOSITY = 5


def analyze_distances(values, sigs):
  """Analyzes distances between signatures."""
  assert len(values) == len(sigs)
  sig_to_value = {}
  num_same_signature = 0
  num_avoidable_same_signature = 0
  num_printed = collections.defaultdict(int)
  for value, sig in zip(values, sigs):
    key = str(sig)
    if key in sig_to_value:
      num_same_signature += 1
      value_type = value.type
      match = sig_to_value[key]
      avoidable = (set(str(v) for v in value.values)
                   != set(str(v) for v in match.values))
      if avoidable:
        num_avoidable_same_signature += 1
      if avoidable and num_printed[value_type] < VERBOSITY:
        print(f'Values of type {value_type} have the same signature:')
        print(f'  {match}')
        print(f'  {value}')
        num_printed[value_type] += 1
    else:
      sig_to_value[key] = value

  print(f'Number of values with already-seen signatures: {num_same_signature} '
        f'({num_same_signature / len(values) * 100:.2f}%)')
  print(f'Number of values with already-seen signatures but avoidable: '
        f'{num_avoidable_same_signature} '
        f'({num_avoidable_same_signature / len(values) * 100:.2f}%)')


def evaluate_property_signatures(value_set, output_value, profile=False):
  """Evaluates how different property signatures are for the set of values."""
  print('\n' + '=' * 80)
  print('\nPerforming evaluation on property signatures.')
  concrete_values = [x for x in value_set if not x.num_free_variables]
  lambda_values = [x for x in value_set if x.num_free_variables]
  print(f'value_set has {len(value_set)} elements:\n'
        f'  * {len(concrete_values)} concrete values\n'
        f'  * {len(lambda_values)} lambda values')
  print()

  start_time = timeit.default_timer()
  concrete_sigs = [
      property_signatures.property_signature_value(v, output_value)
      for v in concrete_values]
  elapsed_time = timeit.default_timer() - start_time
  print(f'Computed {len(concrete_sigs)} signatures for concrete values in '
        f'{elapsed_time:.3f} total seconds, or '
        f'{elapsed_time/len(concrete_sigs):.5f} seconds each.')
  print()

  start_time = timeit.default_timer()
  if profile:
    pr = cProfile.Profile()
    pr.enable()
    lambda_sigs = [
        property_signatures.property_signature_value(v, output_value)
        for v in lambda_values]
    pr.disable()
    pr.print_stats(sort='cumtime')
  else:
    lambda_sigs = [
        property_signatures.property_signature_value(v, output_value)
        for v in lambda_values]
  elapsed_time = timeit.default_timer() - start_time
  print(f'Computed {len(lambda_sigs)} signatures for lambda values in '
        f'{elapsed_time:.3f} total seconds, or '
        f'{elapsed_time/len(lambda_sigs):.5f} seconds each.')
  print()

  print(f'Signature length: {len(lambda_sigs[0])}')
  assert all(len(s) == property_signatures.LAMBDA_SIGNATURE_LENGTH
             for s in lambda_sigs)
  assert all(len(s) == property_signatures.CONCRETE_SIGNATURE_LENGTH
             for s in concrete_sigs)
  print()

  lambda_functionality_dict = collections.defaultdict(list)
  for v, sig in zip(lambda_values, lambda_sigs):
    io_pairs_per_example = property_signatures.run_lambda(v)
    if io_pairs_per_example and len(io_pairs_per_example) == 1:
      # "Broadcast" constant values across the number of examples, so that
      # `IsEven(0)` -> [True] is treated as the same functionality as
      # `IsEven(in1)` -> [True, True] if there are 2 examples, both of which
      # have in1 being even.
      io_pairs_per_example *= output_value.num_examples
    key = str(io_pairs_per_example)
    lambda_functionality_dict[key].append((v, sig))

  failure_key = 'None'
  if failure_key in lambda_functionality_dict:
    num_failed = len(lambda_functionality_dict[failure_key])
    print(f'Of the lambda values, {num_failed} fail to execute.')
    print(f'We are ignoring these and keeping the other '
          f'{len(lambda_values) - num_failed} values that do execute.')
    print(f'  Example value that fails to execute: '
          f'{lambda_functionality_dict[failure_key][0][0]}')
    del lambda_functionality_dict[failure_key]

  lambda_values = []
  lambda_sigs = []
  num_duplicate_functionality = 0
  num_printed = 0
  for key, values in lambda_functionality_dict.items():
    first_value, first_sig = values[0]
    lambda_values.append(first_value)
    lambda_sigs.append(first_sig)
    num_duplicate_functionality += len(values) - 1
    if len(values) > 1 and num_printed < VERBOSITY:
      print(f'The following lambdas have the same functionality: {key[:100]}')
      for v, _ in values[:VERBOSITY]:
        print(f'  {v}')
      num_printed += 1
  print(f'Found {num_duplicate_functionality} lambdas with duplicate '
        f'functionality.')
  print(f'We are ignoring those and keeping the other {len(lambda_values)} '
        f'values with unique functionality.')
  print()

  print('For concrete signatures:')
  analyze_distances(concrete_values, concrete_sigs)
  print()

  print('For lambda signatures:')
  analyze_distances(lambda_values, lambda_sigs)
  print()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  inputs_dict = {
      'delta': [3, 5, -1, 6],
      'lst': [[1, 2, 3], [4, 5, 6], [2, 7], [9, 5, 2, 8, 5]],
  }
  outputs = [
      [4, 5, 6],
      [9, 10, 11],
      [1, 6],
      [15, 11, 8, 14, 11],
  ]
  # outputs = [
  #     [4, 5],
  #     [9, 10],
  #     [1, 6],
  #     [15, 11],
  # ]
  task = task_module.Task(inputs_dict, outputs, solution='')
  domain = domains.get_domain('deepcoder')
  result, value_set, _, _ = baseline_enumeration.synthesize_baseline(
      task, domain, timeout=60)
  print(f'Found solution: {result.expression()}')
  assert (result.expression() ==
          'Map(lambda u1: (lambda v1: Add(delta, v1))(u1), lst)')
          # 'Take(-1, Map(lambda u1: (lambda v1: Add(delta, v1))(u1), lst))')

  value_set = sorted(list(value_set), key=str)

  evaluate_property_signatures(value_set, value_module.OutputValue(outputs),
                               profile=False)


if __name__ == '__main__':
  app.run(main)

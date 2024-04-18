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

"""Defines the Operations used in search."""

import abc

from crossbeam.algorithm import variables as variables_module
from crossbeam.dsl import value as value_module


def comma_variable_list(variables):
  tokens = [', '] * (2 * len(variables) - 1)
  tokens[0::2] = [v.name for v in variables]
  return tokens


class OperationBase(abc.ABC):
  """An operation used in synthesis."""
  invention_counter = 0
  def __init__(self, name, arity, weight=1, num_bound_variables=None, inv_name = None):
    self.name = name
    self.arity = arity
    self.weight = weight
    if self.name == 'Invented':
      self.name = inv_name
    else:
      self.name = name
      
    if num_bound_variables is None:
      self.num_bound_variables = [0] * self.arity
      self.bound_variables = [[]] * self.arity
    else:
      self.num_bound_variables = num_bound_variables
      self.bound_variables = [[value_module.get_bound_variable(i)
                               for i in range(num_bound)]
                              for num_bound in num_bound_variables]
    assert len(self.bound_variables) == arity
    self.has_any_bound_variables = bool(sum(self.bound_variables, []))

  def __hash__(self):
    return hash(repr(self))

  def __eq__(self, other):
    return repr(self) == repr(other)

  def __repr__(self):
    return self.name

  def arg_types(self):
    """The types of this operation's arguments, or None to allow any types."""
    return None

  def return_type(self):
    """The return type of this operation."""
    return None

  def apply(self, arg_values, arg_variables=None, free_variables=None):
    """Applies the operation to a list of arguments, for all examples."""
    num_examples = max(v.num_examples for v in arg_values)
    arg_types = self.arg_types()  # pylint: disable=assignment-from-none
    if arg_types is not None and arg_types != tuple(x.type for x in arg_values):
      return None

    if arg_variables is None:
      arg_variables = [[]] * self.arity

    easy_case = (not self.has_any_bound_variables and
                 not any(x for x in arg_variables) and
                 not free_variables)
    if easy_case:
      try:
        results = [self.apply_single([value[i] for value in arg_values])
                   for i in range(num_examples)]
      except Exception as e:  # pylint: disable=broad-except
        # test = [self.apply_single([value[i] for value in arg_values]) for i in range(num_examples)]
        return None

    else:
      code_parts = []
      if free_variables:  # The final result is a lambda.
        # Make sure the free_variables are in canonical order. That is,
        # `lambda v2, v1: Op(v2, v1)` is disallowed because it's equivalent to
        # `lambda v1, v2: Op(v1, v2)`.
        if free_variables != variables_module.first_free_vars(
            len(free_variables)):
          return None
        code_parts.append(f'lambda {",".join(v.name for v in free_variables)}:')

      code_parts.append('apply([')
      locals_dicts = [{'__builtins__': {}, 'apply': self.apply_single}
                      for _ in range(num_examples)]

      for i, (arg_value, variables, bound_variables) in enumerate(zip(
          arg_values, arg_variables, self.bound_variables)):

        # Make sure arg variables are only free or bound variables, and not
        # input variables. That is, `(lambda v1: Op(v1))(x1)` is disallowed
        # because it's equivalent to `Op(x1)`.
        if not set(variables).issubset(variables_module.ARGV_SET):
          return None

        if i > 0:
          code_parts.append(',')

        if isinstance(arg_value, value_module.FreeVariable):
          code_parts.append(arg_value.name)

        else:
          assert len(variables) == len(arg_value.free_variables)
          arg_name = f'arg_{i}'
          for example_index in range(num_examples):
            locals_dicts[example_index][arg_name] = arg_value[example_index]
          for v in variables:
            if (isinstance(v, value_module.InputVariable) and
                v.name not in locals_dicts):
              for example_index in range(num_examples):
                locals_dicts[example_index][v.name] = v[example_index]

          if bound_variables:  # This argument is a lambda required by the op.
            code_parts.append(
                f'lambda {",".join(v.name for v in bound_variables)}:')
          code_parts.append(arg_name)
          if variables:  # This argument is computed by calling a lambda.
            code_parts.append(f'({",".join(v.name for v in variables)})')

      code_parts.append('])')
      code = ''.join(code_parts)

      try:
        results = [eval(code, locals_dicts[i]) for i in range(num_examples)]  # pylint: disable=eval-used
      except Exception as e:  # pylint: disable=broad-except
        # Some exception occured in apply_single. This is ok, just throw out
        # this value.
        # FIXME: for higher-order macros we end up here
        return None

    value = value_module.OperationValue(results, self, arg_values,
                                        arg_variables, free_variables)
    if self.return_type() is not None:
      value.type = self.return_type()  # pylint: disable=assignment-from-none
    return value

  @abc.abstractmethod
  def apply_single(self, raw_arg_values):
    """Applies the operation to a list of arguments, for 1 example."""

  def expression(self, arg_values, arg_variables, free_variables):
    """Returns a code expression for an application of this operation."""
    return ''.join(self.tokenized_expression(arg_values, arg_variables,
                                             free_variables))

  def tokenized_expression(self, arg_values, arg_variables, free_variables):
    """Returns a tokenized expression for an application of this operation."""
    tokens = []
    if free_variables:
      tokens.append('(lambda ')
      tokens.extend(comma_variable_list(free_variables))
      tokens.append(': ')

    tokens.extend([self.name, '('])
    for i, (arg, variables, bound_variables) in enumerate(
        zip(arg_values, arg_variables, self.bound_variables)):
      if i > 0:
        tokens.append(', ')
      if bound_variables:
        tokens.append('(lambda ')
        tokens.extend(comma_variable_list(bound_variables))
        tokens.append(': ')
      if variables:
        tokens.append('(')
        tokens.extend(arg.tokenized_expression())
        tokens.extend([')', '('])
        tokens.extend(comma_variable_list(variables))
        tokens.append(')')
      else:
        tokens.extend(arg.tokenized_expression())
      if bound_variables:
        tokens.append(')')
    tokens.append(')')
    if free_variables:
      tokens.append(')')
    return tokens

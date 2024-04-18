"""Running and simplifying DeepCoder solutions."""

import functools
import re

from crossbeam.dsl import deepcoder_operations

_OPS_NAMESPACE = {
    op.name: functools.partial(lambda *args, op: op.apply_single(args), op=op)
    for op in deepcoder_operations.get_operations()
}

_OPS_NAMESPACE_LAMBDA = {
    op.name: functools.partial(lambda *args, op: op.apply_single(args), op=op)
    for op in deepcoder_operations.get_operations_()
}

def get_ops_namespace(dreamcoder_prims = True):
   if dreamcoder_prims:
      return _OPS_NAMESPACE
   else:
      return _OPS_NAMESPACE_LAMBDA


def run_program(program, inputs_dict, dreamcoder_prims = True, inventions = None):
  outputs = []
  if inventions is not None:
    inv_namespace = {
      op.name: functools.partial(lambda *args, op: op.apply_single(args), op=op)
      for op in inventions
    }
  else:
    inv_namespace = {}
  num_examples = len(inputs_dict[next(iter(inputs_dict))])
  for i in range(num_examples):
    namespace = get_ops_namespace(False).copy()
    if inv_namespace is not None or inv_namespace != {}:
        namespace = {**namespace, **inv_namespace}
    namespace.update({name: value[i] for name, value in inputs_dict.items()})
    outputs.append(eval(program, namespace))  # pylint: disable=eval-used
  return outputs


def tokenize(s):
    return re.findall(r'[()]|[^()]+', s)

def parse(toks, depth=0):
    ast = []

    while toks:
        t = toks.pop(0)

        if t == '(':
            ast.append(parse(toks, depth + 1))
        elif t == ')':
            if not depth:
                raise SyntaxError('mismatched )')
            return ast
        else:
            ast.append(t)

    if depth:
        raise SyntaxError('premature EOF')

    return ast

def generate(el):
    if not isinstance(el, list):
        return el

    while len(el) == 1 and isinstance(el[0], list):
        el = el[0]

    return '(' + ''.join(map(generate, el)) + ')'



def simplify(program, verbose=False):
  """Replace `(lambda a: expr(a))(b)` with `expr(b)`."""
  program = re.sub(r'\s+', ' ', program.strip())
  program = re.sub(r'\s*\(\s*', '(', program)
  program = re.sub(r',\(', ', (', program)
  program = re.sub(r'\s*\)\s*', ')', program)
  program = re.sub(r':(?=[^ ])', ': ', program)
  if verbose:
    print(program)

  changed = True
  while changed:  # Loop until there are no more changes.
    changed = False

    # One could do this with an AST walking approach, but we'd need third-party
    # tools to turn the AST back into code. Instead, let's just do it manually
    # with string parsing. Note that regex can't match balanced parentheses, so
    # we count those manually too.
    matching_paren_index = [None] * len(program)
    open_paren_index_stack = []
    for i, c in enumerate(program):
      if c == '(':
        open_paren_index_stack.append(i)
      elif c == ')':
        match_index = open_paren_index_stack.pop()
        matching_paren_index[match_index] = i
        matching_paren_index[i] = match_index
    assert not open_paren_index_stack

    # We want to find open parens followed by `lambda`, where the matching
    # closing paren is immediately followed by another pair of parens with a
    # variable list inside (and nothing else). We want to process the innermost
    # lambdas first, which will always appear after its outer lambdas, so we
    # start looking for this pattern from the end.
    for i, c in reversed(list(enumerate(program))):
      if c != '(':
        continue
      open_1 = i
      if not program[open_1 + 1:].startswith('lambda '):
        continue
      close_1 = matching_paren_index[i]
      open_2 = close_1 + 1
      if open_2 >= len(program) or program[open_2] != '(':
        continue
      close_2 = matching_paren_index[open_2]
      part_1 = program[open_1 + 1:close_1]
      match_1 = re.fullmatch(r'lambda ((?:\w+, )*\w+): (.*)', part_1)
      if not match_1:
        continue
      arg_names = match_1.group(1).split(', ')
      expression = match_1.group(2)
      if verbose:
        print(f'arg_names: {arg_names}')
        print(f'expression: {expression}')
      part_2 = program[open_2 + 1:close_2]
      match_2 = re.fullmatch(r'(?:\w+, )*\w+', part_2)
      if not match_2:
        continue
      variable_names = match_2.group(0).split(', ')
      assert len(arg_names) == len(variable_names)
      renaming_dict = dict(zip(arg_names, variable_names))

      all_tokens = re.findall(r'\w+|\W+', expression)
      for token_i, token in enumerate(all_tokens):
        if token in renaming_dict:
          all_tokens[token_i] = renaming_dict[token]
      new_expression = ''.join(all_tokens)
      if verbose:
        print(f'variable_names: {variable_names}')
        print(f'renaming_dict: {renaming_dict}')
        print(f'new_expression: {new_expression}')

      program = program[:open_1] + new_expression + program[close_2 + 1:]
      changed = True
      break  # Once we change one thing, the matching paren indices are stale.

  return program

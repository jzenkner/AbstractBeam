"""Operations for the DeepCoder domain."""
# pylint: disable=unidiomatic-typecheck

from crossbeam.dsl import operation_base


def deepcoder_small_value_filter(x):
  """Checks whether a single value is ok for DeepCoder."""
  if x is None:
    return False
  if isinstance(x, int):
    return -256 <= x <= 255
  if isinstance(x, list):
    return (len(x) <= 20 and
            # isinstance(False, int) is True. We don't want booleans in lists.
            all(type(e) is int  # pylint: disable=unidiomatic-typecheck
                and deepcoder_small_value_filter(e)
                for e in x))
  return True


class DeepCoderOperation(operation_base.OperationBase):
  """A base class for DeepCoder operations."""

  def __init__(self, *args, **kwargs):
    super(DeepCoderOperation, self).__init__(self.__class__.__name__,
                                             *args, **kwargs)


################################################################################
# First-order functions returning int.
################################################################################


class Add(DeepCoderOperation):

  def __init__(self):
    super(Add, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) not in (int, str) or type(right) not in (int, str):
      return None
    return left + right


class Subtract(DeepCoderOperation):

  def __init__(self):
    super(Subtract, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left - right


class Multiply(DeepCoderOperation):

  def __init__(self):
    super(Multiply, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left * right


class IntDivide(DeepCoderOperation):

  def __init__(self):
    super(IntDivide, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left // right


class Square(DeepCoderOperation):

  def __init__(self):
    super(Square, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:
      return None
    return x ** 2


class Min(DeepCoderOperation):

  def __init__(self):
    super(Min, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return min(left, right)


class Max(DeepCoderOperation):

  def __init__(self):
    super(Max, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return max(left, right)


################################################################################
# First-order functions taking or returning bool.
################################################################################


class Greater(DeepCoderOperation):

  def __init__(self):
    super(Greater, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left > right


class Less(DeepCoderOperation):

  def __init__(self):
    super(Less, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) is not int or type(right) is not int:
      return None
    return left < right


class Equal(DeepCoderOperation):

  def __init__(self):
    super(Equal, self).__init__(2)

  def apply_single(self, raw_args):
    left, right = raw_args
    if type(left) not in [int, bool] or type(left) != type(right):
      return None
    return left == right


class IsEven(DeepCoderOperation):

  def __init__(self):
    super(IsEven, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:
      return None
    return x % 2 == 0


class IsOdd(DeepCoderOperation):

  def __init__(self):
    super(IsOdd, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    if type(x) is not int:
      return None
    return x % 2 == 1


class If(DeepCoderOperation):

  def __init__(self):
    super(If, self).__init__(3)

  def apply_single(self, raw_args):
    condition, x, y = raw_args
    if type(condition) is not bool or type(x) is not int or type(y) is not int:
      return None
    return x if condition else y


################################################################################
# First-order functions manipulating lists (returning list or an element).
################################################################################


class Head(DeepCoderOperation):

  def __init__(self):
    super(Head, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    return x[0]


class Last(DeepCoderOperation):

  def __init__(self):
    super(Last, self).__init__(1)

  def apply_single(self, raw_args):
    x = raw_args[0]
    return x[-1]


class Take(DeepCoderOperation):

  def __init__(self):
    super(Take, self).__init__(2)

  def apply_single(self, raw_args):
    n, xs = raw_args
    if type(n) is not int:
      return None
    return xs[:n]


class Drop(DeepCoderOperation):

  def __init__(self):
    super(Drop, self).__init__(2)

  def apply_single(self, raw_args):
    n, xs = raw_args
    if type(n) is not int:
      return None
    return xs[n:]


class Access(DeepCoderOperation):
  """DeepCoder's Access operation."""

  def __init__(self):
    super(Access, self).__init__(2)

  def apply_single(self, raw_args):
    n, xs = raw_args
    # DeepCoder chooses to error if n is negative; we use Python's negative
    # indexing convention (our DSL is a superset of DeepCoder's anyway).
    if type(n) is not int:
      return None
    return xs[n]


class Minimum(DeepCoderOperation):

  def __init__(self):
    super(Minimum, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return min(xs)


class Maximum(DeepCoderOperation):

  def __init__(self):
    super(Maximum, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return max(xs)


class Reverse(DeepCoderOperation):

  def __init__(self):
    super(Reverse, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return list(reversed(xs))


class Sort(DeepCoderOperation):

  def __init__(self):
    super(Sort, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return sorted(xs)


class Sum(DeepCoderOperation):

  def __init__(self):
    super(Sum, self).__init__(1)

  def apply_single(self, raw_args):
    xs = raw_args[0]
    return sum(xs)


################################################################################
# Higher-order functions.
################################################################################


class Map(DeepCoderOperation):

  def __init__(self):
    super(Map, self).__init__(2, num_bound_variables=[1, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    return list(map(f, xs))


class Filter(DeepCoderOperation):
  """DeepCoder's Filter operation."""

  def __init__(self):
    super(Filter, self).__init__(2, num_bound_variables=[1, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    conditions = [f(x) for x in xs]
    if not all(isinstance(x, bool) for x in conditions):
      return None
    return [x for x, c in zip(xs, conditions) if c]


class Count(DeepCoderOperation):
  """DeepCoder's Count operation."""

  def __init__(self):
    super(Count, self).__init__(2, num_bound_variables=[1, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    conditions = [f(x) for x in xs]
    if not all(isinstance(x, bool) for x in conditions):
      return None
    return sum(conditions)


class ZipWith(DeepCoderOperation):

  def __init__(self):
    super(ZipWith, self).__init__(3, num_bound_variables=[2, 0, 0])

  def apply_single(self, raw_args):
    f, xs, ys = raw_args
    return [f(x, y) for x, y in zip(xs, ys)]


class Scanl1(DeepCoderOperation):
  """DeepCoder's Scanl1 operation."""

  def __init__(self):
    super(Scanl1, self).__init__(2, num_bound_variables=[2, 0])

  def apply_single(self, raw_args):
    f, xs = raw_args
    ys = [xs[0]]
    for n in range(1, len(xs)):
      y = f(ys[n-1], xs[n])
      if not deepcoder_small_value_filter(y):
        # Through repeated squaring, scanl1 can produce absurdly large integers
        # causing timeouts if not caught here.
        return None
      ys.append(y)
    return ys


def get_operations():
  return [
      Add(),
      Subtract(),
      Multiply(),
      IntDivide(),
      Square(),
      Min(),
      Max(),
      Greater(),
      Less(),
      Equal(),
      IsEven(),
      IsOdd(),
      If(),
      Head(),
      Last(),
      Take(),
      Drop(),
      Access(),
      Minimum(),
      Maximum(),
      Reverse(),
      Sort(),
      Sum(),
      Map(),
      Filter(),
      Count(),
      ZipWith(),
      Scanl1(),
  ]

"""Handwritten DeepCoder-style tasks using lambdas."""

import collections
import re

from crossbeam.data.deepcoder import solution_weight
from crossbeam.dsl import task as task_module


Task = task_module.Task

# pylint: disable=line-too-long
HANDWRITTEN_TASKS = [

    # Tasks without higher-order functions.
    Task(
        name='none:deepcoder_0',
        inputs_dict={
            'k': [2, 2, 3, 6, 5],
            'b': [[3, 5, 4, 7, 5],
                  [12, 56, 31, 23, 49],
                  [8, 3, 4, 12, 5, 2, 4],
                  [9, 7, 5, 4, 12, 6, 2, 6],
                  [9, 0, 12, -4, 3, -7, 39, 5, 1, -3]],
        },
        outputs=[7, 35, 9, 30, -13],
        solution='Sum(Take(k, Sort(b)))',
    ),
    Task(
        name='none:hailstone',
        inputs_dict={
            'x': [8, 26, 5, 7, 23],
        },
        outputs=[4, 13, 16, 22, 70],
        solution='If(IsEven(x), IntDivide(x, 2), Add(1, Multiply(3, x)))',
    ),
    Task(
        name='none:coins',
        inputs_dict={
            'x': [3, 8, 5, 11, 14],
            'y': [5, 13, 14, 7, 9],
        },
        outputs=[7, 83, 51, 59, 103],
        solution='Subtract(Subtract(Multiply(x, y), x), y)',
    ),
    Task(
        name='none:next_multiple',
        inputs_dict={
            'x': [3, 8, 13, 27, 58],
            'k': [5, 7, 4, 9, 14],
        },
        outputs=[5, 14, 16, 36, 70],
        solution='Multiply(Add(IntDivide(x, k), 1), k)',
    ),
    Task(
        name='none:drop_last',
        inputs_dict={
            'x': [[5, 3, 2, 1, 6],
                  [4, -11, 5, -6, -5, 7, 8],
                  [13, 48, 28, 37, 85, 43, 23, 54, 42, 17]],
            'n': [2, 5, 3],
        },
        outputs=[[5, 3, 2], [4, -11], [13, 48, 28, 37, 85, 43, 23]],
        solution='Take(Subtract(0, n), x)',
    ),
    Task(
        name='none:decreasing_suffix',
        inputs_dict={
            'x': [[3, 2, 5, 7],
                  [4, 6, 2, 1, 7, 3, 9],
                  [-78, 49, 28, -42, -13, 76, 84, -49, 31, 8]],
            'n': [2, 5, 4],
        },
        outputs=[[7, 5],
                 [9, 7, 3, 2, 1],
                 [84, 31, 8, -49]],
        solution='Reverse(Sort(Take(n, Reverse(x))))',
    ),
    Task(
        name='none:remove_ends',
        inputs_dict={
            'x': [[4, 2, 5, 6],
                  [6, -4, 4, -7, 1, 0, -2],
                  [42, 68, 38, 22, 87, 17, 83, 29]],
        },
        outputs=[[2, 5], [-4, 4, -7, 1, 0], [68, 38, 22, 87, 17, 83]],
        solution='Take(-1, Drop(1, x))',
    ),
    Task(
        name='none:slice',
        inputs_dict={
            'x': [[6, 7, 2, 8, 1, 3],
                  [24, 63, 74, 12, 33, 42, 19],
                  [-6, 49, -24, 8, 47, 23, 11, -12, -55, 38]],
            'i': [3, 2, 5],
            'j': [5, 6, 8],
        },
        outputs=[[8, 1],
                 [74, 12, 33, 42],
                 [23, 11, -12]],
        solution='Drop(i, Take(j, x))',
    ),
    Task(
        name='none:top_k',
        inputs_dict={
            'x': [[4, 1, 2, 6, 3],
                  [8, 9, 3, 5, 2, 7, 8],
                  [12, 63, 21, 59, 32, 55, 68, -99, 48]],
            'k': [2, 4, 3],
        },
        outputs=[[6, 4], [9, 8, 8, 7], [68, 63, 59]],
        solution='Take(k, Reverse(Sort(x)))',
    ),
    Task(
        name='none:kth_largest',
        inputs_dict={
            'x': [[5, 9, 4],
                  [7, 8, 6, 2],
                  [18, 38, 24, 74, 37, 87, 64],
                  [-5, 0, -4, 4, 7, -5, 7, -5, 6],
                  [2, -10, 0, -3, -5, 7, 9, -8, 7, 4]],
            'k': [0, 2, 4, 3, 7],
        },
        outputs=[9, 6, 37, 4, -5],
        solution='Access(k, Reverse(Sort(x)))',
    ),
    Task(
        name='none:outer_average',
        inputs_dict={
            'x': [[59, 37, 23],
                  [58, 32, 47, 16],
                  [-44, 45, 38, -63, -12, 18],
                  [-8, -4, 18, 12, -5, -6, 2, 0, -14],
                  [35, 85, 53, 68, 47, 17, 86, 27, 75, 91]],
        },
        outputs=[41, 37, -13, -11, 63],
        solution='IntDivide(Add(Head(x), Last(x)), 2)',
    ),
    Task(
        name='none:triple_median',
        inputs_dict={
            'x': [[9, 7, 4],
                  [23, 54, 38],
                  [-6, 5, -7],
                  [5, 8, 13],
                  [-47, -18, -21]],
        },
        outputs=[7, 38, -6, 8, -21],
        solution='Access(1, Sort(x))',
    ),
    Task(
        name='none:range',
        inputs_dict={
            'x': [[16, -14, 5, 22, -7],
                  [43, 58, 13, 29, 67, 38],
                  [4, 5, 3, 4, 8, 3, 6, 5, 6],
                  [26, 24, 28, 27, 28, 29, 24, 22, 26, 25],
                  [42, -35, 48, -57, 42, 13, 58, 62, -32, -49]],
        },
        outputs=[36, 54, 5, 7, 119],
        solution='Subtract(Maximum(x), Minimum(x))',
    ),
    Task(
        name='none:range_both',
        inputs_dict={
            'x': [[58],
                  [5, 2, 6, 4],
                  [64, 24, 81, 57, 25, 73],
                  [67, 92, 21, 13, 57, 48],
                  [63, -35, 68, -54, -23, 9, 18, -34, -26]],
            'y': [[25],
                  [9, 4, 3],
                  [41, 32, 53, 18, 27, 20, 61],
                  [43, 17, 58, 34, 23],
                  [25, 61, -58, -2, -62, 77, 23, 18, 47]]
        },
        outputs=[33, 7, 63, 79, 139],
        solution='Subtract(Max(Maximum(x), Maximum(y)), Min(Minimum(x), Minimum(y)))',
    ),
    Task(
        name='none:slope',
        inputs_dict={
            'x': [[9, 2],
                  [-3, 1],
                  [37, 32],
                  [-17, -14],
                  [6, -3]],
            'y': [[58, 23],
                  [45, -79],
                  [-15, 50],
                  [-46, -4],
                  [5, 77]],
        },
        outputs=[5, -31, -13, 14, -8],
        solution='IntDivide(Subtract(Last(y), Head(y)), Subtract(Last(x), Head(x)))',
    ),

    # Tasks primarily using Map.
    ############################
    Task(
        name='map:cube',
        inputs_dict={
            'x': [[3], [4, 1, 2], [-1, 5, 0, -4, 2, 3, -2]],
        },
        outputs=[[27], [64, 1, 8], [-1, 125, 0, -64, 8, 27, -8]],
        solution='Map((lambda u1: Multiply(u1, Square(u1))), x)',
    ),
    Task(
        name='map:absolute_value',
        inputs_dict={
            'x': [[8, -4, 2, -3, -7, 6],
                  [-23, 31, -12, -46, 78, 0, -32, -61, 48],
                  [98, -58, 43, -42, -68, 47, -15, -21, 74, -29]],
        },
        outputs=[[8, 4, 2, 3, 7, 6],
                 [23, 31, 12, 46, 78, 0, 32, 61, 48],
                 [98, 58, 43, 42, 68, 47, 15, 21, 74, 29]],
        solution='Map((lambda u1: Max(u1, Subtract(0, u1))), x)',
    ),
    Task(
        name='map:multiply_previous',
        inputs_dict={
            'x': [[6, 3, 4],
                  [-4, 6, 2, -6, 7],
                  [3, -3, -5, 2, 0, 1, -4, 8, -6]],
        },
        outputs=[[30, 6, 12],
                 [20, 30, 2, 42, 42],
                 [6, 12, 30, 2, 0, 0, 20, 56, 42]],
        solution='Map((lambda u1: Subtract(Square(u1), u1)), x)',
    ),
    Task(
        name='map:triangular',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                  [0, 5, 2, 1, 6],
                  [7, 2, 5, 9, 2, 0, 4, 1, 3]],
        },
        outputs=[[1, 3, 6, 10],
                 [0, 15, 3, 1, 21],
                 [28, 3, 15, 45, 3, 0, 10, 1, 6]],
        solution='Map((lambda u1: IntDivide(Add(Square(u1), u1), 2)), x)',
    ),
    Task(
        name='map:linear_equation',
        inputs_dict={
            'x': [[14, 23, -16],
                  [4, 2, 6, 0, 3],
                  [3, -7, 4, 1, 5, -2, 0, 1, 2, -1]],
            'a': [-3, 4, 7],
            'b': [25, -7, 10],
        },
        outputs=[[-17, -44, 73],
                 [9, 1, 17, -7, 5],
                 [31, -39, 38, 17, 45, -4, 10, 17, 24, 3]],
        solution='Map((lambda u1: Add(Multiply(a, u1), b)), x)',
    ),
    Task(
        name='map:linear_transform',
        inputs_dict={
            'x': [[20, -10, 0],
                  [4, 2, 3, 1, 5],
                  [7, 9, -6, 12, 11, 4, 6, 4]],
        },
        outputs=[[56, -34, -4],
                 [8, 2, 5, -1, 11],
                 [17, 23, -22, 32, 29, 8, 14, 8]],
        solution='Map((lambda u1: Subtract(Multiply(u1, 3), 4)), x)',
    ),
    Task(
        name='map:quadratic_transform',
        inputs_dict={
            'x': [[3, 5, 1, 6],
                  [7, -1, 4, 0, -2, 3],
                  [5, -3, 9, -10, 11, 2, 7, 10, -8]],
        },
        outputs=[[15, 35, 3, 48],
                 [63, -1, 24, 0, 0, 15],
                 [35, 3, 99, 80, 143, 8, 63, 120, 48]],
        solution='Map((lambda u1: Multiply(u1, Add(u1, 2))), x)',
    ),
    Task(
        name='map:clip',
        inputs_dict={
            'x': [[5, -4, 6, 17, -3, 7],
                  [53, 71, 28, 61, 27, 93, 57, 74],
                  [16, -4, -5, 14, 13, -7, -6, 17, -3, 15]],
            'a': [0, 52, -5],
            'b': [9, 64, 15],
        },
        outputs=[[5, 0, 6, 9, 0, 7],
                 [53, 64, 52, 61, 52, 64, 57, 64],
                 [15, -4, -5, 14, 13, -5, -5, 15, -3, 15]],
        solution='Map((lambda u1: Max(Min(u1, b), a)), x)',
    ),
    Task(
        name='map:square_keep_sign',
        inputs_dict={
            'x': [[4, -6, 8, -7],
                  [-1, -4, 6, -5, 2],
                  [7, 9, -3, 4, -2, 0, -4, -8, 5, 1]],
        },
        outputs=[[16, -36, 64, -49],
                 [-1, -16, 36, -25, 4],
                 [49, 81, -9, 16, -4, 0, -16, -64, 25, 1]],
        solution='Map((lambda u1: Multiply(If(Less(u1, 0), -1, 1), Square(u1))), x)',
    ),
    Task(
        name='map:sort_square',
        inputs_dict={
            'x': [[3, 4, 5], [2, 6, -3], [3, 1, -1, 6, 0, 3, 2, 7, -5]],
        },
        outputs=[[9, 16, 25], [4, 9, 36], [0, 1, 1, 4, 9, 9, 25, 36, 49]],
        solution='Sort(Map((lambda u1: Square(u1)), x))',
    ),
    Task(
        name='map:zero_min',
        inputs_dict={
            'x': [[3, 7, 4, 9],
                  [16, 10, 25, 16, 83],
                  [-4, 3, 1, 0, -5, 5]],
        },
        outputs=[[0, 4, 1, 6],
                 [6, 0, 15, 6, 73],
                 [1, 8, 6, 5, 0, 10]],
        solution='Map((lambda u1: Subtract(u1, Minimum(x))), x)',
    ),
    Task(
        name='map:zero_element',
        inputs_dict={
            'x': [[7, 3, 8],
                  [25, 74, 34, 42, 60],
                  [-6, 8, 2, 5, -7, 1, 4, -3, 0, 9]],
            'i': [1, 3, 7],
        },
        outputs=[[4, 0, 5],
                 [-17, 32, -8, 0, 18],
                 [-3, 11, 5, 8, -4, 4, 7, 0, 3, 12]],
        solution='Map((lambda u1: Subtract(u1, Access(i, x))), x)',
    ),
    Task(
        name='map:shift_first_to',
        inputs_dict={
            'x': [[7, 1, 4, 8, 3],
                  [4, 14, 11, 8, 18, 11, 9, 3],
                  [2, -5, 7, 11, -3, 2, 15, -15, -13]],
            't': [2, 10, -5],
        },
        outputs=[[2, -4, -1, 3, -2],
                 [10, 20, 17, 14, 24, 17, 15, 9],
                 [-5, -12, 0, 4, -10, -5, 8, -22, -20]],
        solution='Map((lambda u1: Add(u1, Subtract(t, Head(x)))), x)',
    ),
    Task(
        name='map:gather',
        inputs_dict={
            'x': [[4, 2, 6, 8],
                  [11, 3, 20, -5, 7],
                  [3, -6, 4, -5, 3, 9, -2, 0, 1, -4]],
            'i': [[0, 2, 1, 3],
                  [1, 4, 2, 1, 3, 1, 4, 1],
                  [2, 0, 2, 8, 4, 2, 3, 5]],
        },
        outputs=[[4, 6, 2, 8],
                 [3, 7, 20, 3, -5, 3, 7, 3],
                 [4, 3, 4, 1, 3, 4, -5, 9]],
        solution='Map((lambda u1: Access(u1, x)), i)',
    ),
    Task(
        name='map:max_selected',
        inputs_dict={
            'x': [[3, 8, 4, 2, 6, 5],
                  [-8, -4, 7, 6, 3, -5, 0, -2, 2, 1],
                  [64, 83, 27, 18, 59, 17, 45, 23, 63],
                  [64, 83, 27, 18, 59, 17, 45, 23, 63],
                  [15, -32, 38, 67, -21, 41, 25, -43, -28, 33]],
            'i': [[3, 2, 0, 4],
                  [5, 4, 1],
                  [2, 3, 4, 6, 7],
                  [2, 3, 5, 6, 7],
                  [7, 2, 1, 0, 8, 5, 2, 0, 1, 2]],
        },
        outputs=[6, 3, 59, 45, 41],
        solution='Maximum(Map((lambda u1: Access(u1, x)), i))',
    ),
    Task(
        name='map:zero_odds',
        inputs_dict={
            'x': [[4, 2, 5, 6, 3],
                  [15, 14, 7, 26, 24, 21],
                  [-42, 25, -39, 0, 34, 12, 55, -14, 21]],
        },
        outputs=[[4, 2, 0, 6, 0],
                 [0, 14, 0, 26, 24, 0],
                 [-42, 0, 0, 0, 34, 12, 0, -14, 0]],
        solution='Map((lambda u1: If(IsOdd(u1), 0, u1)), x)',
    ),
    Task(
        name='map:replace',
        inputs_dict={
            'x': [[7, 2, 4, 6, 4, 2, 5],
                  [-6, -3, 4, 3, -5, -3, 2, 1, 5],
                  [18, 48, 27, 26, 27, 27, 28, 17, 27, 33]],
            'f': [4, -3, 27],
            'r': [-1, 7, 99],
        },
        outputs=[[7, 2, -1, 6, -1, 2, 5],
                 [-6, 7, 4, 3, -5, 7, 2, 1, 5],
                 [18, 48, 99, 26, 99, 99, 28, 17, 99, 33]],
        solution='Map((lambda u1: If(Equal(u1, f), r, u1)), x)',
    ),
    Task(
        name='map:replace_negative',
        inputs_dict={
            'x': [[7, -6, 8, -3, -5, 2],
                  [-6, -4, 2, -3, 14, -15, 2, -5, -8],
                  [13, 63, -23, -35, 32, 45, -78, -31, 12, -14]],
            'y': [-9, 5, 99],
        },
        outputs=[[7, -9, 8, -9, -9, 2],
                 [5, 5, 2, 5, 14, 5, 2, 5, 5],
                 [13, 63, 99, 99, 32, 45, 99, 99, 12, 99]],
        solution='Map((lambda u1: If(Less(u1, 0), y, u1)), x)',
    ),
    Task(
        name='map:spread_last',
        inputs_dict={
            'x': [[4, 2, 6, 3, 5],
                  [-7, 17, -4, -12, 3, -9],
                  [41, 23, 58, 21, 14, 63, 28, 36]],
        },
        outputs=[[9, 7, 11, 8],
                 [-16, 8, -13, -21, -6],
                 [77, 59, 94, 57, 50, 99, 64]],
        solution='Take(-1, Map((lambda u1: Add(u1, Last(x))), x))',
    ),
    Task(
        name='map:fill_first',
        inputs_dict={
            'x': [[3, 6, 1, 4, 7],
                  [-26, 27, 51, -14, 31, -17, -32],
                  [73, 67, 48, 27, 97, 85, 37, 68, 21, 84]],
        },
        outputs=[[3, 3, 3, 3, 3],
                 [-26, -26, -26, -26, -26, -26, -26],
                 [73, 73, 73, 73, 73, 73, 73, 73, 73, 73]],
        solution='Map((lambda u1: Head(x)), x)',
    ),
    Task(
        name='map:length',
        inputs_dict={
            'x': [[65, 24],
                  [9, 45, 2, 16, 4, 2, 63],
                  [8, -4, 28, 27, -32, 23, 15, 2, 0],
                  [0],
                  [-1, 3, 1, 2, 4, -2, 6, -4, -8, 3]],
        },
        outputs=[2, 7, 9, 1, 10],
        solution='Sum(Map((lambda u1: 1), x))',
    ),
    Task(
        name='map:median',
        inputs_dict={
            'x': [[6, 5, 2],
                  [42, 86, 31, 58, 75],
                  [18, 2, 34, 7, 0, 56, 32, 48, 54],
                  [18, 2, 34, 37, 0, 56, 32, 48, 54],
                  [85, 48, -42, -15, -25, 23, -52]],
        },
        outputs=[5, 58, 32, 34, -15],
        solution='Access(IntDivide(Sum(Map((lambda u1: 1), x)), 2), Sort(x))',
    ),
    Task(
        name='map:is_greater',
        inputs_dict={
            'x': [[19, 13, 17, 18, 20],
                  [5, 7, 4, 5, 6, 2, 3],
                  [-9, -6, 2, -7, 18, -11, -8, 1, -15, -4]],
            'y': [17, 4, -7],
        },
        outputs=[[1, 0, 0, 1, 1],
                 [1, 1, 0, 1, 1, 0, 0],
                 [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]],
        solution='Map((lambda u1: If(Greater(u1, y), 1, 0)), x)',
    ),
    Task(
        name='map:divide_even',
        inputs_dict={
            'x': [[8, 5, 6, 14, 9],
                  [42, -15, 67, 0, -16, 23, 31, 28],
                  [62, 24, 68, -44, -25, -14, 2, 5, -20, 41]],
        },
        outputs=[[4, 5, 3, 7, 9],
                 [21, -15, 67, 0, -8, 23, 31, 14],
                 [31, 12, 34, -22, -25, -7, 1, 5, -10, 41]],
        solution='Map((lambda u1: If(IsEven(u1), IntDivide(u1, 2), u1)), x)',
    ),
    Task(
        name='map:multiply_odd',
        inputs_dict={
            'x': [[3, 1, 2, 5, 4, 7],
                  [15, 21, 28, 7, 33, 26, 42, 27, 18],
                  [7, -22, -1, 28, 9, -17, 5, 13, -4, 11]],
            'n': [3, 2, -3],
        },
        outputs=[[9, 3, 2, 15, 4, 21],
                 [30, 42, 28, 14, 66, 26, 42, 54, 18],
                 [-21, -22, 3, 28, -27, 51, -15, -39, -4, -33]],
        solution='Map((lambda u1: If(IsOdd(u1), Multiply(u1, n), u1)), x)',
    ),

    # Tasks primarily using Filter.
    ###############################
    Task(
        name='filter:greater',
        inputs_dict={
            'x': [[1, 3, 4, 2],
                  [6, 4, 3, 5, 9, 2],
                  [25, 0, 79, -1, -45, 31, -4, 7, -2, 11]],
            'y': [2, 4, -3],
        },
        outputs=[[3, 4], [6, 5, 9], [25, 0, 79, -1, 31, 7, -2, 11]],
        solution='Filter((lambda u1: Greater(u1, y)), x)',
    ),
    Task(
        name='filter:sort_even',
        inputs_dict={
            'x': [[7, 6, 2, 5, 8, 0, 5],
                  [52, 35, 68, 37, 41, 22, 64, 57, 26],
                  [16, -24, 28, -29, -12, -17, 23, 58, 21, -14]],
        },
        outputs=[[0, 2, 6, 8],
                 [22, 26, 52, 64, 68],
                 [-24, -14, -12, 16, 28, 58]],
        solution='Filter((lambda u1: IsEven(u1)), Sort(x))',
    ),
    Task(
        name='filter:max_odd',
        inputs_dict={
            'x': [[32, 57, 60, 53],
                  [7, 12, 6, 18, 13, 14, 5, 6],
                  [-6, 8, 5, -7, -13, 1, -4, 18, 32],
                  [32, 25, 68, 43, 42, 44, 56, 37, 28, 29],
                  [32, 25, 68, 43, 42, 44, 55, 37, 28, 29]],
        },
        outputs=[57, 13, 5, 43, 55],
        solution='Maximum(Filter((lambda u1: IsOdd(u1)), x))',
    ),
    Task(
        name='filter:odd_negative',
        inputs_dict={
            'x': [[-11, 7, -3, -8, 2],
                  [2, -5, 3, -4, -7, -1, -2],
                  [-17, 23, -40, -25, -31, 42, 0, -23, -7, -16]],
        },
        outputs=[[-11, -3],
                 [-5, -7, -1],
                 [-17, -25, -31, -23, -7]],
        solution='Filter((lambda u1: IsOdd(Min(u1, 0))), x)',
    ),
    Task(
        name='filter:sum_positive',
        inputs_dict={
            'x': [[-5, 8, -10, 7, 2],
                  [16, 21, -14, 21, 1, -49, 31],
                  [45, -45, -38, -41, -6, 45, -13, 20],
                  [-46, -32, -29, 33, -3, -13, 9, 23, -39],
                  [-4, 8, 23, 15, 8, -31, -7, -9, -9, 24]],
        },
        outputs=[17, 90, 110, 65, 78],
        solution='Sum(Filter((lambda u1: Greater(u1, 0)), x))',
    ),
    Task(
        name='filter:negative_descending',
        inputs_dict={
            'x': [[-4, 8, 1, -2, -9, -5],
                  [-11, 10, -13, -19, 0, -12, -10],
                  [3, -38, -48, 32, -43, -19, 23, -36, -44, 48]],
        },
        outputs=[[-2, -4, -5, -9],
                 [-10, -11, -12, -13, -19],
                 [-19, -36, -38, -43, -44, -48]],
        solution='Filter((lambda u1: Less(u1, 0)), Reverse(Sort(x)))',
    ),
    Task(
        name='filter:between',
        inputs_dict={
            'x': [[7, 6, 1, 4, 9, 3, 5, 8],
                  [11, 78, 73, 35, 70, 59, 16, 74, 24],
                  [-12, -21, 37, 20, 13, 21, 19, 0, -19, -20]],
            'a': [3, 15, -20],
            'b': [8, 74, 20],
        },
        outputs=[[7, 6, 4, 5],
                 [73, 35, 70, 59, 16, 24],
                 [-12, 13, 19, 0, -19]],
        solution='Filter((lambda u1: Less(u1, b)), Filter((lambda u1: Greater(u1, a)), x))',
    ),
    Task(
        name='filter:multiple',
        inputs_dict={
            'x': [[24, 19, 22, 16, 10, 44],
                  [21, -35, -25, 0, -17, -14, 15, 42],
                  [11, 84, 79, 16, -57, 97, 27, -23, -39, -32]],
            'n': [4, 7, 3],
        },
        outputs=[[24, 16, 44],
                 [21, -35, 0, -14, 42],
                 [84, -57, 27, -39]],
        solution='Filter((lambda u1: Equal(u1, Multiply(IntDivide(u1, n), n))), x)',
    ),
    Task(
        name='filter:find_factors',
        inputs_dict={
            'x': [[4, 5, 2, 15, 3, 5, 1, 7],
                  [4, 5, 6, 2, 3, 7, 9, 1, 8],
                  [-5, 7, -8, 12, -4, 2, -1, 4, 6, -10]],
            'n': [45, 28, 40],
        },
        outputs=[[5, 15, 3, 5, 1],
                 [4, 2, 7, 1],
                 [-5, -8, -4, 2, -1, 4, -10]],
        solution='Filter((lambda u1: Equal(n, Multiply(IntDivide(n, u1), u1))), x)',
    ),
    Task(
        name='filter:fixpoints',
        inputs_dict={
            'x': [[0, 5, 4, 3, 5, 5, 6, 1],
                  [3, 1, 2, 4, 6, 5, 8, 0, 7],
                  [0, 9, 2, 3, 4, 7, 6, 8, 7, 9]],
        },
        outputs=[[0, 3, 5, 5, 5, 6],
                 [1, 2, 5],
                 [0, 2, 3, 4, 6, 9, 9]],
        solution='Sort(Filter((lambda u1: Equal(u1, Access(u1, x))), x))',
    ),
    Task(
        name='filter:find_at',
        inputs_dict={
            'x': [[8, 7, 4, 7, 7, 2],
                  [17, -9, -10, -9, -9, 23, -9],
                  [42, 43, 41, 42, 45, 48, 42, 42, 49, 42]],
            'y': [7, -9, 42],
            'i': [[0, 1, 2, 4, 5],
                  [0, 1, 0, 4, 6, 2, 1, 5, 4, 1],
                  [6, 2, 5, 4, 1, 3, 0, 9, 7, 8]],
        },
        outputs=[[1, 4],
                 [1, 4, 6, 1, 4, 1],
                 [6, 3, 0, 9, 7]],
        solution='Filter((lambda u1: Equal(Access(u1, x), y)), i)',
    ),

    # Tasks primarily using Count.
    ##############################
    Task(
        name='count:num_match',
        inputs_dict={
            'x': [[5, 7, 8, 7, 9, 6],
                  [5, 7, 8, 7, 9, 6],
                  [7, 0, 7, 0, 7, 7],
                  [42, 34, 42, 42, 42, 56, 42, 38, 42, 42],
                  [42, 34, 42, 42, 38, 56, 42, 38, 42, 42]],
            'n': [7, 8, 7, 42, 42],
        },
        outputs=[2, 1, 4, 7, 6],
        solution='Count((lambda u1: Equal(u1, n)), x)',
    ),
    Task(
        name='count:rank',
        inputs_dict={
            'x': [[28, 43, 69, 38, 42],
                  [35, -54, 48, 17, 49, -24, 32],
                  [-7, -4, 8, -5, -4, -3, -8, 4],
                  [2, 6, 0, 3, 1, 4, 2, 3, 5, 0],
                  [2, 6, 0, 5, 1, 4, 2, 3, 5, 0]],
            'q': [44, 18, -3, 4, 4],
        },
        outputs=[4, 3, 5, 7, 6],
        solution='Count((lambda u1: Less(u1, q)), x)',
    ),
    Task(
        name='count:num_under_first',
        inputs_dict={
            'x': [[5, 2, 6, 7, 1, 3, 8],
                  [26, 0, 19, 22, 39, 47, 1, 26],
                  [65, 33, 16, 53, 34, 95, 86, 4, 12],
                  [-11, 23, -48, -24, -11, -4, 44, 7, -26, -12],
                  [-11, 23, -48, -24, -11, -14, 44, 7, -26, -12]],
        },
        outputs=[3, 4, 6, 4, 5],
        solution='Count((lambda u1: Less(u1, Head(x))), x)',
    ),
    Task(
        name='count:count_negative',
        inputs_dict={
            'x': [[6, 7, 0, -4, 1, -5, 2],
                  [11, -4, -1, 19, -5, 18, -8, -13],
                  [25, 16, -15, 18, -3, -12, 1, 29, -21],
                  [-18, 18, 46, -7, -15, -44, -33, 8, -1, 11],
                  [32, 42, -46, -97, 53, 34, 8, -9, 36, 20]],
        },
        outputs=[2, 5, 4, 6, 3],
        solution='Count((lambda u1: Less(u1, 0)), x)',
    ),
    Task(
        name='count:count_last',
        inputs_dict={
            'x': [[11, 10, 9, 7, 10, 19, 10],
                  [47, 23, 44, 3, 23, 40, 23, 44],
                  [47, 23, 44, 3, 23, 40, 23, 23],
                  [-2, 26, -20, -19, 23, -29, -9, 2, -23],
                  [-42, 0, -42, -42, -41, -43, -42, 42, -42, -42]],
        },
        outputs=[3, 2, 4, 1, 6],
        solution='Count((lambda u1: Equal(u1, Last(x))), x)',
    ),
    Task(
        name='count:count_at',
        inputs_dict={
            'x': [[14, 11, 28, 11, -17, -1, -21],
                  [24, 33, 24, 24, 23, 0, 24, 25],
                  [26, -7, -21, 20, -26, -7, -29, -7, -23],
                  [48, -21, -97, 3, 88, -21, -21, -20, -21, -21],
                  [48, -21, -97, 3, 88, -21, -21, -21, -21, -21]],
            'i': [3, 6, 1, 7, 7]
        },
        outputs=[2, 4, 3, 1, 6],
        solution='Count((lambda u1: Equal(u1, Access(i, x))), x)',
    ),
    Task(
        name='count:even_or_positive',
        inputs_dict={
            'x': [[6, -5, -1, -6, 0, -3, -4],
                  [6, -5, -2, -6, 0, -3, -4],
                  [6, -5, -2, -6, 0, 3, -4],
                  [-9, -3, -7, -8, -11, -53, -32, -5, 23, -11],
                  [-4, -7, -6, 3, 0, -9, -12, 4, 5, -3]],
        },
        outputs=[4, 5, 6, 3, 7],
        solution='Count((lambda u1: IsEven(Min(u1, 0))), x)',
    ),
    Task(
        name='count:total_zeros',
        inputs_dict={
            'x': [[-2, 0, 3, 1],
                  [8, 4, 0, 1, 0, 2],
                  [0, 1, 0],
                  [17, 0, -2, 0, -11, -9, -6],
                  [0, 0, 0, 42, 0, 0, 0, 63, 0, -14]],
            'y': [[0, 0, 1, -1],
                  [0, 3, 0, 0, 5, 9],
                  [6, 0, 6, 0, 0, 0, 5],
                  [-11, 3, 0, -11, -15, 0, -15, -12],
                  [0, -54, 0, 14, 35, 0, 0]]
        },
        outputs=[3, 5, 6, 4, 11],
        solution='Add(Count((lambda u1: Equal(u1, 0)), x), Count((lambda u1: Equal(u1, 0)), y))',
    ),
    Task(
        name='count:min_costs',
        inputs_dict={
            'r': [[4, 0, 7, 3, 0],
                  [-5, 2, -1, 0, 5, 1, 3, -6],
                  [1, 0, 3, 2, 0, 4, 0, 2, 3, 4]],
            'c': [[6, 8, 2, 5, 9],
                  [11, 25, 14, 7, 16, 5, 10, 18, 10],
                  [61, 32, 44, 67, 17, 73, 25, 68, 95, 43]],
        },
        outputs=[[2, 5, 6],
                 [5, 7, 10, 10],
                 [17, 25, 32, 43, 44, 61, 67]],
        solution='Take(Count((lambda u1: Greater(u1, 0)), r), Sort(c))',
    ),

    # Tasks primarily using ZipWith.
    ################################
    Task(
        name='zipwith:deepcoder_1',
        inputs_dict={
            'w': [[6, 2, 4, 7, 9],
                  [10, 30, 20],
                  [8, 6, 5, 9],
                  [7, 5, 1, 6, 2, 7],
                  [14, 7, 2, 9, 3, 13, 10, 11]],
            't': [[5, 3, 6, 1, 0],
                  [65, 7, 32],
                  [2, 5, 9, 1],
                  [0, 9, 4, 5, 7, 2],
                  [1, 4, 2, 11, 7, 5, 7, 4]],
        },
        outputs=[27, 97, 28, 24, 44],
        solution='Maximum(ZipWith((lambda u1, u2: Add(Multiply(u1, 3), u2)), w, t))',
        original_solution='Maximum(ZipWith(lambda u1, u2: Add(u1, u2), Map(lambda u1: Multiply(u1, 3), w), t))',
    ),
    Task(
        name='zipwith:deepcoder_4',
        inputs_dict={
            'x': [[7, 3, 8, 2, 5],
                  [5, 6, 1],
                  [3, 5, 1, 2],
                  [7, 2, 1, 9, 3, 6],
                  [4, 2, 6, 19, 5, 13, 2, 3]],
            'y': [[2, 8, 9, 1, 3],
                  [8, 2, 3],
                  [4, 6, 1, 3],
                  [5, 1, 6, 4, 7, 5],
                  [2, 9, 3, 11, 1, 5, 16, 15]],
        },
        outputs=[79, 35, 28, 101, 219],
        solution='Sum(ZipWith((lambda u1, u2: Multiply(u1, u2)), Sort(x), Reverse(Sort(y))))',
    ),
    Task(
        name='zipwith:deepcoder_5',
        inputs_dict={
            'a': [[3, 7, 5, 2, 8],
                  [14, 11, 15, 18, 14, 12],
                  [85, 3, 35, 1, 74, 47, 2, 26, 33, 43]],
        },
        outputs=[[3, 2, 5, 2, 3],
                 [12, 11, 15, 15, 11, 12],
                 [43, 3, 26, 1, 47, 47, 1, 26, 3, 43]],
        solution='ZipWith((lambda u1, u2: Min(u1, u2)), a, Reverse(a))',
    ),
    Task(
        name='zipwith:deepcoder_6',
        inputs_dict={
            't': [[4, 8, 11, 2],
                  [46, 24, 68],
                  [8, 22, 18, 6],
                  [21, 26, 10, 11, 15],
                  [11, 18, 15, 33, 18, 23, 7]],
            'p': [[2, 3, 4, 1],
                  [35, 37, 31],
                  [26, 19, 21, 38],
                  [3, 11, 22, 10, 9],
                  [29, 23, 24, 7, 14, 12, 25]]
        },
        outputs=[1, 59, 32, 19, 30],
        solution='Subtract(Minimum(ZipWith((lambda u1, u2: Add(u1, u2)), t, p)), 2)',
        original_solution='Minimum(ZipWith(lambda u1, u2: Add(u1, u2), Map(lambda u1: Subtract(u1, 1), t), Map(lambda u1: Subtract(u1, 1), p)))',
    ),
    Task(
        name='zipwith:deepcoder_8',
        inputs_dict={
            's': [[1, 2, 4, 5, 7],
                  [2, 4, 11, 26],
                  [14, 36, 54, 89],
                  [1, 7, 10, 11, 13, 18, 19],
                  [4, 5, 8, 9, 10, 11, 12, 15, 17, 18]],
        },
        outputs=[9, 31, 93, 32, 37],
        solution='Sum(ZipWith((lambda u1, u2: Max(Subtract(u1, u2), 0)), Reverse(s), s))',
        original_solution='Sum(Filter(lambda u1: Greater(u1, 0), ZipWith(lambda u1, u2: Subtract(u1, u2), Reverse(s), s)))',
    ),
    Task(
        name='zipwith:dot',
        inputs_dict={
            'x': [[2, 0],
                  [3, 1, 2],
                  [7, 8, 1, 4],
                  [2, -4, 6, -5, -1, 3, 2],
                  [-4, 0, 4, 6, 2, -3, 1, 5, 6, 5]],
            'y': [[10, 5],
                  [8, 2, -5],
                  [1, 0, 4, 2],
                  [3, 8, 4, 4, 2, 0, 5],
                  [2, 1, 0, 2, 0, 6, 2, 1, 3, 2]],
        },
        outputs=[20, 16, 19, -14, 21],
        solution='Sum(ZipWith((lambda u1, u2: Multiply(u1, u2)), x, y))',
    ),
    Task(
        name='zipwith:average',
        inputs_dict={
            'x': [[8, 0], [12, 3, 7], [-5, 4, 3, -7, 11, 0]],
            'y': [[2, 4], [4, 1, 7], [3, 6, -3, -11, 19, -6]],
        },
        outputs=[[5, 2], [8, 2, 7], [-1, 5, 0, -9, 15, -3]],
        solution='ZipWith((lambda u1, u2: IntDivide(Add(u1, u2), 2)), x, y)',
    ),
    Task(
        name='zipwith:three_sum',
        inputs_dict={
            'a': [[6, 3], [2, 1, 5], [4, 2, 1, 5, 3]],
            'b': [[2, 7], [6, -4, 2], [8, 4, 0, 2, 1]],
            'c': [[0, 0], [3, 0, -3], [0, 1, 8, 0, 6]],
        },
        outputs=[[8, 10], [11, -3, 4], [12, 7, 9, 7, 10]],
        solution='ZipWith((lambda u1, u2: Add(u1, u2)), a, ZipWith((lambda u1, u2: Add(u1, u2)), b, c))',
    ),
    Task(
        name='zipwith:sort_divide',
        inputs_dict={
            'm': [[43, 68, 27],
                  [28, 41, 57, 52],
                  [78, 68, 25, 76, 67, 38, 49, 74, 47]],
            'n': [[5, 7, 13],
                  [6, 2, 6, 4],
                  [10, 4, 19, 12, 19, 7, 7, 15, 3]],
        },
        outputs=[[2, 8, 9],
                 [4, 9, 13, 20],
                 [1, 3, 4, 5, 6, 7, 7, 15, 17]],
        solution='Sort(ZipWith((lambda u1, u2: IntDivide(u1, u2)), m, n))',
    ),
    Task(
        name='zipwith:pair_differences',
        inputs_dict={
            'x': [[16, 43, 64],
                  [34, 13, 23, 58, 46],
                  [9, 49, -27, -37, 26, -38, 9, -12, 19]]
        },
        outputs=[[27, 21],
                 [-21, 10, 35, -12],
                 [40, -76, -10, 63, -64, 47, -21, 31]],
        solution='ZipWith((lambda u1, u2: Subtract(u2, u1)), x, Drop(1, x))',
    ),
    Task(
        name='zipwith:sum_mask',
        inputs_dict={
            'x': [[6, 11, 2, 7],
                  [4, 5, 9, 0, 7, 9],
                  [3, 13, 17, 5, 0, 14, 18],
                  [2, -12, -5, 9, -13, -2, -9, 5],
                  [-38, 34, 20, 32, 10, -10, 4, -1, -27]],
            'm': [[1, 0, 1, 1],
                  [0, 1, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1],
                  [1, 0, 0, 1, 0, 1, 1, 0, 0]],
        },
        outputs=[15, 21, 31, -20, -12],
        solution='Sum(ZipWith((lambda u1, u2: Multiply(u1, u2)), x, m))',
    ),
    Task(
        name='zipwith:distance',
        inputs_dict={
            'x': [[2, 6, 1, 7, 4],
                  [-6, 3, 15, -4, 11, -1, 7, -9],
                  [15, 62, 47, 23, 50, 13, 52, 28, 58]],
            'y': [[4, 1, 5, 4, 8],
                  [3, 6, -2, 0, -8, -3, 4, 2],
                  [63, 44, 78, 65, 13, 31, 23, 22, 68]],
        },
        outputs=[[2, 5, 4, 3, 4],
                 [9, 3, 17, 4, 19, 2, 3, 11],
                 [48, 18, 31, 42, 37, 18, 29, 6, 10]],
        solution='ZipWith((lambda u1, u2: Max(Subtract(u1, u2), Subtract(u2, u1))), x, y)',
    ),
    Task(
        name='zipwith:squared_distance',
        inputs_dict={
            'x': [[26],
                  [6, 14],
                  [8, 5, 7],
                  [-3, -2, 9, 4],
                  [-6, 2, -5, 4, -2]],
            'y': [[35],
                  [9, 8],
                  [6, 9, 8],
                  [-5, 1, 6, -2],
                  [-1, -2, 1, 4, 7]],
        },
        outputs=[81, 45, 21, 58, 158],
        solution='Sum(ZipWith((lambda u1, u2: Square(Subtract(u1, u2))), x, y))',
    ),
    Task(
        name='zipwith:is_equal',
        inputs_dict={
            'x': [[5, 2, 7, 3, 5],
                  [0, -2, 6, 4, 5, -4],
                  [-2, -3, 2, 9, 4, -1, 4, 7, 5]],
            'y': [[5, 1, 7, 3, 6],
                  [0, -2, 0, -4, 5, -4],
                  [5, -2, 2, 9, 3, -1, -2, 7, 7]],
        },
        outputs=[[1, 0, 1, 1, 0],
                 [1, 1, 0, 0, 1, 1],
                 [0, 0, 1, 1, 0, 1, 0, 1, 0]],
        solution='ZipWith((lambda u1, u2: If(Equal(u1, u2), 1, 0)), x, y)',
    ),
    Task(
        name='zipwith:sorted_min',
        inputs_dict={
            'x': [[7, 2, 5, 6],
                  [4, -7, -6, 7, 1, -6],
                  [12, 20, 27, 19, 19, 29, 23, 7, 1]],
            'y': [[4, 8, 1, 9],
                  [-8, 2, -6, 0, 9, -1],
                  [26, 18, 25, 0, 9, 28, 19, 22, 17]],
        },
        outputs=[[1, 4, 6, 7],
                 [-8, -6, -6, 0, 2, 7],
                 [0, 7, 12, 18, 19, 20, 23, 26, 28]],
        solution='ZipWith((lambda u1, u2: Min(u1, u2)), Sort(x), Sort(y))',
    ),
    Task(
        name='zipwith:add_outer_triples',
        inputs_dict={
            'x': [[3, 2, 6, 8, 1],
                  [4, 3, 7, 6, 2, 5],
                  [-4, 16, -12, -5, 7, 0, -2, -5, -3, -7]],
        },
        outputs=[[9, 10, 7],
                 [10, 5, 12],
                 [-9, 13, -19]],
        solution='ZipWith((lambda u1, u2: Add(u1, u2)), x, Reverse(Take(3, Reverse(x))))',
    ),
    Task(
        name='zipwith:max_mirror_product',
        inputs_dict={
            'x': [[3, 9, 4, 2, 5],
                  [2, 3, 1, 8, 3, 5],
                  [-4, 5, -3, 5, -7, 3, -3],
                  [-4, 5, -3, 5, -7, 3, -3, 4],
                  [1, 0, 3, 1, 2, 4, 0, 4, 2, 1]],
        },
        outputs=[18, 10, 25, -9, 12],
        solution='Maximum(ZipWith((lambda u1, u2: Multiply(u1, u2)), x, Reverse(x)))',
    ),
    Task(
        name='zipwith:double_or_nothing',
        inputs_dict={
            'p': [[5, 6, 8, 2, 6],
                  [10, 15, 30, 20, 25, 40, 5],
                  [22, 53, 31, 47, 27, 24, 14, 39, 42, 13]],
            'w': [[1, 1, -1, 1, -1],
                  [-1, 1, -1, -1, 1, -1, 1],
                  [1, -1, -1, 1, -1, 1, 1, -1, -1, 1]],
        },
        outputs=[[10, 12, 0, 4, 0],
                 [0, 30, 0, 0, 50, 0, 10],
                 [44, 0, 0, 94, 0, 48, 28, 0, 0, 26]],
        solution='ZipWith((lambda u1, u2: Multiply(u1, Add(1, u2))), p, w)',
    ),

    # Tasks primarily using Scanl1.
    ###############################
    Task(
        name='scanl1:deepcoder_3',
        inputs_dict={
            'h': [[8, 5, 7, 2, 5],
                  [8, 5, 7, 1, 5],
                  [6, 4, 8, 2, 7, 3],
                  [6, 7, 8, 6, 7, 5, 2, 5, 7, 4],
                  [10, 8, 9, 9, 6, 8, 10, 5, 2, 7]],
        },
        outputs=[5, 6, 10, 14, 13],
        solution='Subtract(Sum(h), Sum(Scanl1((lambda u1, u2: Min(u1, u2)), h)))',
        original_solution='Sum(Filter(lambda u1: Greater(u1, 0), ZipWith(lambda u1, u2: Subtract(u1, u2), h, Scanl1(lambda u1, u2: Min(u1, u2), h))))',
    ),
    Task(
        name='scanl1:running_max',
        inputs_dict={
            'x': [[1, 6, 3, 20],
                  [4, 2, 6, 3, 1, 7, 3, 9],
                  [-6, -5, -2, 0, -1, 3, 3, 2, 5, 4]],
        },
        outputs=[[1, 6, 6, 20],
                 [4, 4, 6, 6, 6, 7, 7, 9],
                 [-6, -5, -2, 0, 0, 3, 3, 3, 5, 5]],
        solution='Scanl1((lambda u1, u2: Max(u1, u2)), x)',
    ),
    Task(
        name='scanl1:max_running_sum',
        inputs_dict={
            'x': [[5, 4, -1, 2, -5],
                  [4, 7, -2, -5, 6, -3, 4],
                  [-3, -1, 8, 2, -1, -5, 4, 1],
                  [4, -2, 3, -5, -1, 2, -4, 7, -3, 2],
                  [4, -2, 3, -5, -1, 2, -4, 17, -3, 2]],
        },
        outputs=[10, 11, 6, 5, 14],
        solution='Maximum(Scanl1((lambda u1, u2: Add(u1, u2)), x))',
    ),
    Task(
        name='scanl1:running_sum_reverse',
        inputs_dict={
            'x': [[7, 4, 5],
                  [3, 1, 6, 8, 2],
                  [-4, 13, 9, 0, -3, -8, 5, 2]],
        },
        outputs=[[5, 9, 16],
                 [2, 10, 16, 17, 20],
                 [2, 7, -1, -4, -4, 5, 18, 14]],
        solution='Scanl1((lambda u1, u2: Add(u1, u2)), Reverse(x))',
    ),
    Task(
        name='scanl1:running_sum_restart',
        inputs_dict={
            'x': [[3, 5, 0, 4, 2],
                  [6, 0, 4, 7, 3, 0, 1, 0, 4, 5],
                  [-6, 3, 8, -4, 0, -3, -1, 0, 8, -5]],
        },
        outputs=[[3, 8, 0, 4, 6],
                 [6, 0, 4, 11, 14, 0, 1, 0, 4, 9],
                 [-6, -3, 5, 1, 0, -3, -4, 0, 8, 3]],
        solution='Scanl1((lambda u1, u2: If(Equal(u2, 0), 0, Add(u1, u2))), x)',
    ),
    Task(
        name='scanl1:running_sum_extra',
        inputs_dict={
            'x': [[3, 2, 6, 4],
                  [-6, 4, -2, 3, 8, -3, -9],
                  [12, -21, 15, 7, -5, 10, 11, -5, -28, 8]],
            'n': [1, 3, -2],
        },
        outputs=[[3, 6, 13, 18],
                 [-6, 1, 2, 8, 19, 19, 13],
                 [12, -11, 2, 7, 0, 8, 17, 10, -20, -14]],
        solution='Scanl1((lambda u1, u2: Add(Add(u1, u2), n)), x)',
    ),
    Task(
        name='scanl1:decay',
        inputs_dict={
            'x': [[7, 2, 6, 1, 5],
                  [10, 14, 8, 3, 12, 6, 2, 7],
                  [15, 12, 27, 39, 12, 11, 25, 0, 0, 3]],
        },
        outputs=[[7, 5, 8, 5, 7],
                 [10, 19, 17, 11, 17, 14, 9, 11],
                 [15, 19, 36, 57, 40, 31, 40, 20, 10, 8]],
        solution='Scanl1((lambda u1, u2: Add(IntDivide(u1, 2), u2)), x)',
    ),
    Task(
        name='scanl1:prefix_running_product',
        inputs_dict={
            'x': [[2, 1, 5, 2, 3, 1, 2],
                  [1, 3, 2, 1, 4, 1, 5],
                  [5, -1, -2, 1, -3, 2, -1, -3, -1, 2]],
            'n': [4, 5, 8],
        },
        outputs=[[2, 2, 10, 20],
                 [1, 3, 6, 6, 24],
                 [5, -5, 10, 10, -30, -60, 60, -180]],
        solution='Scanl1((lambda u1, u2: Multiply(u1, u2)), Take(n, x))',
    ),
    Task(
        name='scanl1:product',
        inputs_dict={
            'x': [[3, 6, 5],
                  [2, 1, 4, 2, 3],
                  [3, 4, 1, 5, 2],
                  [5, -2, 1, -1, -3, 1, 2],
                  [-2, 1, 2, 2, -3, -1, 3, -1]],
        },
        outputs=[90, 48, 120, -60, 72],
        solution='Last(Scanl1((lambda u1, u2: Multiply(u1, u2)), x))',
    ),
    Task(
        name='scanl1:running_min_no_ends',
        inputs_dict={
            'x': [[4, 7, 5, 6, 3, 8, 2],
                  [8, 6, 7, 5, 4, 5, 2, 5],
                  [32, 57, 45, 68, 43, -28, -12, -30, -43, -5]],
        },
        outputs=[[7, 5, 5, 3, 3],
                 [6, 6, 5, 4, 4, 2],
                 [57, 45, 45, 43, -28, -28, -30, -43]],
        solution='Scanl1((lambda u1, u2: Min(u1, u2)), Take(-1, Drop(1, x)))',
    ),
    Task(
        name='scanl1:alternate_signs',
        inputs_dict={
            'x': [[3, 1, 5, 7, 3, 4],
                  [-5, 3, 6, -3, -2, 9, 4, -8, 5, -3],
                  [50, 28, -68, 56, 43, 21, -12, 48, 87, 57]],
        },
        outputs=[[3, -1, 5, -7, 3, -4],
                 [-5, 3, -6, 3, -2, 9, -4, 8, -5, 3],
                 [50, -28, 68, -56, 43, -21, 12, -48, 87, -57]],
        solution='Scanl1((lambda u1, u2: If(Greater(Multiply(u1, u2), 0), Subtract(0, u2), u2)), x)',
    ),
    Task(
        name='scanl1:follow_cycle',
        inputs_dict={
            'x': [[4, 0, 2, 3, 1],
                  [3, 5, 0, 1, 6, 4, 2],
                  [8, 4, 7, 5, 6, 0, 9, 1, 3, 2]],
        },
        outputs=[[4, 1, 0, 4, 1],
                 [3, 1, 5, 4, 6, 2, 0],
                 [8, 3, 5, 0, 8, 3, 5, 0, 8, 3]],
        solution='Scanl1((lambda u1, u2: Access(u1, x)), x)',
    ),
    Task(
        name='scanl1:make_strictly_increasing',
        inputs_dict={
            'x': [[2, 4, 3, 7, 8, 8],
                  [-5, -8, -5, 1, -4, 5, 5, 7],
                  [-40, -23, -59, -9, 0, 4, 2, -4, 18, 14]],
        },
        outputs=[[2, 4, 5, 7, 8, 9],
                 [-5, -4, -3, 1, 2, 5, 6, 7],
                 [-40, -23, -22, -9, 0, 4, 5, 6, 18, 19]],
        solution='Scanl1((lambda u1, u2: Max(Add(u1, 1), u2)), x)',
    ),
    Task(
        name='scanl1:closest_to_zero',
        inputs_dict={
            'x': [[7, -5, 6, 4, -7, -2, -1],
                  [-6, 5, 8, -4, -6, 5, 3, 0, -6],
                  [57, 54, -37, 46, -33, -65, 21, -20, -14, 15]],
        },
        outputs=[[7, -5, -5, 4, 4, -2, -1],
                 [-6, 5, 5, -4, -4, -4, 3, 0, 0],
                 [57, 54, -37, -37, -33, -33, 21, -20, -14, -14]],
        solution='Scanl1((lambda u1, u2: If(Less(Square(u1), Square(u2)), u1, u2)), x)',
    ),

    # Tasks using a combination of higher-order functions.
    ######################################################
    Task(
        name='multi:deepcoder_2',
        inputs_dict={
            'a': [[6, 2, 4, 7, 9],
                  [7, 2, 4, 5, 7, 8, 4, 3],
                  [7, 2, 4, 5, 7, 8, 4, 3],
                  [11, 31, 52, 68, 37, 53, 69, 42, 53, 27],
                  [11, 31, 52, 68, 37, 53, 69, 42, 53, 27]],
            'b': [[5, 3, 2, 1, 0],
                  [8, 1, 4, 5, 3, 5, 1, 2],
                  [8, 1, 4, 4, 3, 5, 1, 2],
                  [10, 28, 47, 78, 37, 38, 62, 17, 44, 23],
                  [38, 62, 17, 44, 23, 10, 28, 47, 78, 37]],
        },
        outputs=[4, 5, 6, 8, 5],
        solution='Count((lambda u1: Greater(u1, 0)), ZipWith((lambda u1, u2: Subtract(u1, u2)), a, b))',
    ),
    Task(
        name='multi:deepcoder_7',
        inputs_dict={
            's': [[4, 7, 2, 3],
                  [5, 2, 1, 4],
                  [3, 5, 6, 1, 4],
                  [8, 4, 3, 2, 2, 4, 3],
                  [6, 3, 5, 5, 1, 3, 4, 1, 2]],
            'p': [[2, 1, 3, 1],
                  [1, 3, 2, 4],
                  [1, 1, 3, 2, 5],
                  [2, 1, 1, 3, 1, 2, 3],
                  [1, 2, 1, 1, 3, 1, 1, 2, 1]],
        },
        outputs=[62, 59, 98, 149, 173],
        solution='Sum(ZipWith((lambda u1, u2: Multiply(u1, u2)), s, Scanl1((lambda u1, u2: Add(u1, u2)), p)))',
    ),
    Task(
        name='multi:range_like',
        inputs_dict={
            'x': [[3, 6, 2],
                  [-6, -3, 6, 1, 0, 2, -1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        },
        outputs=[[0, 1, 2],
                 [0, 1, 2, 3, 4, 5, 6],
                 [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        solution='Scanl1((lambda u1, u2: Add(u1, 1)), Map((lambda u1: 0), x))',
    ),
    Task(
        name='multi:find_odd_products',
        inputs_dict={
            'x': [[4, 5, 7, 3, 8],
                  [6, -1, 9, 4, -3, 11, 7, -7, 8],
                  [11, 5, 2, 7, 13, 3, 5, 6, 5, 7]],
        },
        outputs=[[35, 21],
                 [-9, -33, 77, -49],
                 [55, 91, 39, 15, 35]],
        solution='Filter((lambda u1: IsOdd(u1)), ZipWith((lambda u1, u2: Multiply(u2, u1)), x, Drop(1, x)))',
    ),
    Task(
        name='multi:multiply_odds',
        inputs_dict={
            'x': [[3, 5, 8, 2, 1],
                  [5, 2, 1, 3, 3, 1, 4],
                  [3, -4, -1, 8, 2, 0, -3, 0, 9, -1]],
        },
        outputs=[[3, 15, 15],
                 [5, 5, 15, 45, 45],
                 [3, -3, 9, 81, -81]],
        solution='Scanl1((lambda u1, u2: Multiply(u1, u2)), Filter((lambda u1: IsOdd(u1)), x))',
    ),
    Task(
        name='multi:running_total',
        inputs_dict={
            'x': [[8, 4, 2, 7],
                  [9, -3, 7, 0, 2, -2, 5, -1, 5],
                  [23, -54, 39, 28, -36, -12, 21, 42, -84, 35]],
            'y': [[2, 5, 3, 1],
                  [8, 0, 4, -3, 7, 0, -4, -6, 1],
                  [-52, 32, 18, -43, -32, 73, -28, 32, 56, -24]],
        },
        outputs=[[10, 19, 24, 32],
                 [17, 14, 25, 22, 31, 29, 30, 23, 29],
                 [-29, -51, 6, -9, -77, -16, -23, 51, 23, 34]],
        solution='Scanl1((lambda u1, u2: Add(u1, u2)), ZipWith((lambda u1, u2: Add(u1, u2)), x, y))',
    ),
    Task(
        name='multi:add_from_index',
        inputs_dict={
            'x': [[3, 6, 7],
                  [11, 23, 8, 15, 4],
                  [-5, 7, 23, -4, 0, -12, 3, 2, -8]],
            'y': [[8, 5, 3, 10, 4],
                  [-5, 0, 5],
                  [-20, 14, 8, -4, 22, 0]],
            'i': [[0, 3, 1],
                  [2, 0, 2, 0, 1],
                  [1, 4, 5, 2, 3, 3, 0, 1, 4]]
        },
        outputs=[[11, 16, 12],
                 [16, 18, 13, 10, 4],
                 [9, 29, 23, 4, -4, -16, -17, 16, 14]],
        solution='ZipWith((lambda u1, u2: Add(u1, u2)), x, Map((lambda u1: Access(u1, y)), i))',
    ),
    Task(
        name='multi:positive_differences',
        inputs_dict={
            'x': [[8, 5, 7, 1, 2],
                  [11, 4, 2, 13, 21, 16, 8, 4],
                  [68, 23, 13, 6, 47, 72, 18, 58, 9, 25]],
            'y': [[3, 1, 7, 2, 1],
                  [2, 3, 7, 15, 5, 16, 3, 5],
                  [24, 73, 88, 31, 19, 58, 42, 13, 7, 10]],
        },
        outputs=[[5, 4, 0, 1],
                 [9, 1, 16, 0, 5],
                 [44, 28, 14, 45, 2, 15]],
        solution='Filter((lambda u1: Greater(u1, -1)), ZipWith((lambda u1, u2: Subtract(u1, u2)), x, y))',
    )
]

    # NEW CHATGPT TASKs
    ######################################################
"""Task(
        name='filter_count_positive',
        inputs_dict={
            'x': [[-1, 2, 3, -4, 5],
                [0, -2, -3, 7, 8],
                [1, -1, 1, -1, 1],
                [10, 20, -30, 40, 50]],
        },
        outputs=[3,2,3,4],
        solution='Count((lambda u1: Greater(u1, 0)), x)',
    ),
    Task(
        name='sum_even_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4, 5],
                [10, 15, 20, 25],
                [2, 4, 6, 8, 10],
                [5, 7, 9]],
        },
        outputs = [6, 30, 30, 0],
        solution='Sum(Filter((lambda u1: IsEven(u1)), x))',
    ),
    Task(
        name='average_unique_elements',
        inputs_dict={
            'x': [[1, 2, 2, 3, 4],
                [5, 5, 5, 5, 5],
                [10, 20, 30, 40, 50],
                [-1, -2, -2, -3, -1]],
        },
        outputs=[2, 5, 30, -2],
        solution='IntDivide(Sum(Sort(x)), Count((lambda u1: True), Sort(x)))',
    ), 
    Task(
        name='maximum_difference',
        inputs_dict={
            'x': [[10, 3, 5, 6],
                [1, 2, 3, 4, 5],
                [5, 17, 2, 8],
                [7, 7, 7, 7]],
        },
        outputs=[7, 4, 15, 0],
        solution='Subtract(Maximum(x), Minimum(x))',
    ),
    Task(
        name='count_negative_numbers',
        inputs_dict={
            'x': [[-4, -3, -2, 1, 2, 3],
                [0, -1, -5, 6, -7],
                [-1, -1, -1, -1, -1],
                [1, 2, 3, 4, 5]],
        },
        outputs=[3, 3, 5, 0],
        solution='Count((lambda u1: Less(u1, 0)), x)',
    ),
    Task(
        name='increment_all_elements',
        inputs_dict={
            'x': [[1, 2, 3],
                [-1, 0, 1, 2],
                [5, 6, 7]],
        },
        outputs=[[2, 3, 4],
                [0, 1, 2, 3],
                [6, 7, 8]],
        solution='Map((lambda u1: Add(u1, 1)), x)',
    ),
    Task(
        name='double_even_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [2, 3, 4, 5, 6],
                [10, 15, 20, 25]],
        },
        outputs=[[1, 4, 3, 8],
                [4, 3, 8, 5, 12],
                [20, 15, 40, 25]],
        solution='Map((lambda u1: If(IsEven(u1), Multiply(u1, 2), u1)), x)',
    ),
    Task(
        name='filter_negatives_and_square',
        inputs_dict={
            'x': [[-1, 2, -3, 4],
                [0, -2, 3, -4, 5],
                [-10, 5, -30, 5]],
        },
        outputs=[[4, 16],
                [0, 9, 25],
                [25, 25]],
        solution='Map((lambda u1: Square(u1)), Filter((lambda u1: Greater(u1, -1)), x))',
    ),
    Task(
        name='sum_squares_positive',
        inputs_dict={
            'x': [[-1, 2, 3],
                [4, -5, 6],
                [-7, 8, -9]],
        },
        outputs=[13, 52, 64],
        solution='Sum(Map((lambda u1: Square(u1)), Filter((lambda u1: Greater(u1, 0)), x)))',
    ),
    Task(
        name='max_square_positive',
        inputs_dict={
            'x': [[-10, 2, 3],
                [4, -5, 6],
                [-8, 9, -10]],
        },
        outputs=[9, 36, 81],
        solution='Maximum(Map((lambda u1: Square(u1)), Filter((lambda u1: Greater(u1, 0)), x)))',
    ),
    Task(
        name='sum_even_after_adding_one',
        inputs_dict={
            'x': [[1, 3, 5],
                [2, 4, 6],
                [7, 8, 10]],
        },
        outputs=[12, 0, 8],
        solution='Sum(Filter((lambda u1: IsEven(u1)), Map((lambda u1: Add(u1, 1)), x)))',
    ),
    Task(
        name='avg_positive_after_squaring',
        inputs_dict={
            'x': [[-1, 2, 3],
                [4, -5, 6],
                [0, 8, -9]],
        },
        outputs=[6, 26, 64],
        solution='IntDivide(Sum(Map((lambda u1: Square(u1)), Filter((lambda u1: Greater(u1, 0)), x))), Count((lambda u1: Greater(u1, 0)), x))',
    ),
    Task(
        name='min_square_negative',
        inputs_dict={
            'x': [[-10, 2, -3],
                [-4, 5, -6],
                [-8, -9, 10]],
        },
        outputs=[9, 16, 64],
        solution='Minimum(Map((lambda u1: Square(u1)), Filter((lambda u1: Less(u1, 0)), x)))',
    ),
    Task(
        name='sum_absolute_values',
        inputs_dict={
            'x': [[-1, 2, -3],
                [4, -5, 6],
                [-7, 8, -9]],
        },
        outputs=[2, 10, 8],
        solution='Sum(Map((lambda u1: If(Greater(u1, 0), u1, 0)), x))',
    ),
    Task(
        name='filter_even_multiply_by_2',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        },
        outputs=[[4, 8],
                [12, 16],
                [20, 24]],
        solution='Map((lambda u1: Multiply(u1, 2)), Filter((lambda u1: IsEven(u1)), x))',
    ),
    Task(
        name='count_zeros',
        inputs_dict={
            'x': [[0, 1, 0, 2],
                [3, 0, 4, 0],
                [5, 6, 7]],
        },
        outputs=[2, 2, 0],
        solution='Count((lambda u1: Equal(u1, 0)), x)',
    ),
    Task(
        name='double_odd_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        },
        outputs=[[2, 2, 6, 4],
                [10, 6, 14, 8],
                [18, 10, 22, 12]],
        solution='Map((lambda u1: If(IsOdd(u1), Multiply(u1, 2), u1)), x)',
    ),
    Task(
        name='square_even_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        },
        outputs=[[1, 4, 3, 16],
                [5, 36, 7, 64],
                [9, 100, 11, 144]],
        solution='Map((lambda u1: If(IsEven(u1), Square(u1), u1)), x)',
    ),
    Task(
        name='negate_negative_numbers',
        inputs_dict={
            'x': [[-1, 2, -3, 4],
                [-5, -6, 7, 8],
                [9, -10, 11, -12]],
        },
        outputs=[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        solution='Map((lambda u1: If(Greater(u1, 0), u1, Multiply(u1, -1))), x)',
    ),
    Task(
        name='append_number_to_all',
        inputs_dict={
            'x': [[1, 2, 3],
                [4, 5],
                [6, 7, 8]],
            'n': [9, 9 ,9],
        },
        outputs=[[10, 11, 12],
                [13, 14],
                [15, 16, 17]],
        solution='Map((lambda u1: Add(u1, n)), x)',
    ),
    Task(
        name='increment_and_square',
        inputs_dict={
            'x': [[1, 2, 3],
                [4, 5],
                [6, 7, 8]],
        },
        outputs=[[4, 9, 16],
                [25, 36],
                [49, 64, 81]],
        solution='Map((lambda u1: Square(Add(u1, 1))), x)',
    ),
    Task(
        name='filter_positive_and_double',
        inputs_dict={
            'x': [[-1, 2, 3, -4],
                [0, 5, -6, 7],
                [8, -9, 10, -11]],
        },
        outputs=[[4, 6],
                [10, 14],
                [16, 20]],
        solution='Map((lambda u1: Multiply(u1, 2)), Filter((lambda u1: Greater(u1, 0)), x))',
    ),
    Task(
        name='square_negative_numbers_only',
        inputs_dict={
            'x': [[-2, 3, -4, 5],
                [-6, -7, 8, 9],
                [-10, 11, -12, 13]],
        },
        outputs=[[4, 3, 16, 5],
                [36, 49, 8, 9],
                [100, 11, 144, 13]],
        solution='Map((lambda u1: If(Less(u1, 0), Square(u1), u1)), x)',
    ),
    Task(
        name='increment_if_even',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        },
        outputs=[[1, 3, 3, 5],
                [5, 7, 7, 9],
                [9, 11, 11, 13]],
        solution='Map((lambda u1: If(IsEven(u1), Add(u1, 1), u1)), x)',
    ),
    Task(
        name='negate_even_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        },
        outputs=[[1, -2, 3, -4],
                [5, -6, 7, -8],
                [9, -10, 11, -12]],
        solution='Map((lambda u1: If(IsEven(u1), Subtract(0, u1), u1)), x)',
    ),
    Task(
        name='triple_odd_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
        },
        outputs=[[3, 2, 9, 4],
                [15, 6, 21, 8],
                [27, 10, 33, 12]],
        solution='Map((lambda u1: If(IsOdd(u1), Multiply(u1, 3), u1)), x)',
    ),
    Task(
        name='sum_of_even_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8, 9]],
        },
        outputs=[6, 14],
        solution="Sum(Filter((lambda u1: IsEven(u1)), x))",
    ),
    Task(
        name='maximum_even_number',
        inputs_dict={
            'x': [[1, 3, 5, 8],
                [2, 4, 6, 7, 9]],
        },
        outputs=[8, 6],
        solution="Maximum(Filter((lambda u1: IsEven(u1)), x))",
    ),
    Task(
        name='count_odd_numbers',
        inputs_dict={
            'x': [[1, 2, 3, 4],
                [5, 6, 7, 8, 9]],
        },
        outputs=[2, 3],
        solution="Count((lambda u1: IsOdd(u1)), x)",
    ), 
    Task(
        name='first_three_squared',
        inputs_dict={
            'x': [[2, 3, 4, 13],
                [1, 3, 6, 17, 18]],
        },
        outputs=[[4, 9, 16],
                [1, 9, 36]],
        solution="Map((lambda u1: Square(u1)), Take(3, x))",
    ),
    Task(
        name='subtract_minimum_from_all',
        inputs_dict={
            'x': [[2, 4, 6, 8],
                [1, 3, 5, 7, 9]],
        },
        outputs=[[0, 2, 4, 6],
                [0, 2, 4, 6, 8]],
        solution='Map((lambda u1: Subtract(u1, Minimum(x))), x)',
    ),
    Task(
        name='double_if_greater_than_five',
        inputs_dict={
            'x': [[3, 5, 6, 8],
                [1, 4, 5, 7, 9]],
        },
        outputs=[[3, 5, 12, 16],
                [1, 4, 5, 14, 18]],
        solution='Map((lambda u1: If(Greater(u1, 5), Multiply(u1, 2), u1)), x)',
    )
"""





SYNTHETIC_TASKS = [
    Task(
        name='synthetic:weight_3_function_0',
        inputs_dict={'x1': [[-159, 106, 174, -57, 248], [-138, -159, 174]]},
        outputs=[[1, 1, 1, 1, 1], [1, 1, 1]],
        solution='Map(lambda u1: 1, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_1',
        inputs_dict={
            'x1': [[9, 0, 1, 3], [1, 8, 4, 5], [5, 7, 0, 6]],
            'x2': [2, 8, 6],
        },
        outputs=[[9, 2, 2, 2], [1, 8, 8, 8], [5, 6, 6, 6]],
        solution='Scanl1(lambda u1, u2: x2, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_2',
        inputs_dict={
            'x1': [
                [-16, -8, -3, -12, -37, -4, -19],
                [-8, -30, -15, -28, -2, -36],
                [-34, -5, -39, -32, -2],
                [0, -9, -28, -45, -13, -29, -17],
                [-18, -16, -29, -26],
            ]
        },
        outputs=[-3, -15, -39, -28, -29],
        solution='Access(2, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_3',
        inputs_dict={
            'x1': [
                [-231, -2, 204, 9, -49, 207, -82, 154, -3],
                [127, 135, 85, -56, -192, -215, 106, -164],
                [-101, 111, -187, 37, 7, -122, -213, 238, -18, -150],
                [-128, -53, -194, 68, 97, 134, -178, -173],
                [-172, 22, 228, -113, 153, -97],
            ]
        },
        outputs=[
            [-49, 207, -82, 154, -3],
            [-192, -215, 106, -164],
            [7, -122, -213, 238, -18, -150],
            [97, 134, -178, -173],
            [153, -97],
        ],
        solution='Drop(4, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_4',
        inputs_dict={
            'x1': [
                [71],
                [-234, -233, 18, 213, -64],
                [206, -26, 90, -63],
                [179],
            ],
            'x2': [-100, -186, -188, 237],
        },
        outputs=[[], [], [], [179]],
        solution='Take(x2, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_5',
        inputs_dict={
            'x1': [1, 1, 1, 1],
            'x2': [
                [53],
                [25, 26, 94, 94, 98],
                [48, 49, 64, 35, 78],
                [38, 77, 92],
            ],
        },
        outputs=[[], [26, 94, 94, 98], [49, 64, 35, 78], [77, 92]],
        solution='Drop(x1, x2)',
    ),
    Task(
        name='synthetic:weight_3_function_6',
        inputs_dict={'x1': [[0, 1], [2, 1, 2, 1, 1]]},
        outputs=[[1], [1, 2, 1, 1]],
        solution='Drop(1, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_7',
        inputs_dict={'x1': [[2, 2, 1, 4], [1, -1, 1, 0]]},
        outputs=[[1, 4], [1, 0]],
        solution='Drop(2, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_8',
        inputs_dict={'x1': [[1, 10], [9, 10], [6, 0], [4, 2]]},
        outputs=[[1, 3], [9, 3], [6, 3], [4, 3]],
        solution='Scanl1(lambda u1, u2: 3, x1)',
    ),
    Task(
        name='synthetic:weight_3_function_9',
        inputs_dict={
            'x1': [0, 1, 2, 2, 0],
            'x2': [
                [108, 205, 206, 224, 125, 247, 140, 174, 212, 212],
                [112, 173, 167, 246, 143, 126, 134, 209],
                [112, 231, 117, 158, 171, 164, 180],
                [113, 233, 195, 207, 124],
                [109, 176, 116],
            ],
        },
        outputs=[108, 173, 117, 195, 109],
        solution='Access(x1, x2)',
    ),
    Task(
        name='synthetic:weight_4_function_0',
        inputs_dict={
            'x1': [-36, -13, 10, 21],
            'x2': [[14, -26], [-8, -27], [-35, -39], [18, 10]],
        },
        outputs=[-1, 0, -4, 0],
        solution='IntDivide(Head(x2), x1)',
    ),
    Task(
        name='synthetic:weight_4_function_1',
        inputs_dict={'x1': [34, 14]},
        outputs=[34, 16],
        solution='Max(x1, Square(4))',
    ),
    Task(
        name='synthetic:weight_4_function_2',
        inputs_dict={
            'x1': [
                [2, 9, 8],
                [9, 2, 6, 4, 7, 5, 4],
                [2, 1, 2, 5, 5, 2],
                [1, 7, 4, 4, 1, 2, 7],
            ]
        },
        outputs=[8, 8, 4, 4],
        solution='Multiply(4, Minimum(x1))',
    ),
    Task(
        name='synthetic:weight_4_function_3',
        inputs_dict={'x1': [[0, 1], [2, 1, 2, 1, 1]]},
        outputs=[4, 0],
        solution='IntDivide(4, Sum(x1))',
    ),
    Task(
        name='synthetic:weight_4_function_4',
        inputs_dict={
            'x1': [
                [2, 4, 9, 8, 7, 6],
                [8, 1, 1, 7, 2],
                [8, 4, 7, 9, 4, 6, 5],
                [9, 9, 2, 8, 2],
                [8, 8, 8, 4, 3],
            ],
            'x2': [[65, 144], [47, 112], [109, 198], [55, 35], [197, 208]],
        },
        outputs=[[65, 2], [47, 1], [109, 4], [55, 2], [197, 3]],
        solution='Scanl1(lambda u1, u2: Minimum(x1), x2)',
    ),
    Task(
        name='synthetic:weight_4_function_5',
        inputs_dict={'x1': [[-213, -36], [-13, -37], [-100, 0]]},
        outputs=[251, 52, 102],
        solution='Subtract(2, Sum(x1))',
    ),
    Task(
        name='synthetic:weight_4_function_6',
        inputs_dict={
            'x1': [[-71], [-43, 97, 45, -93, 8, -34]],
            'x2': [[-37, -42, -27, -57, -85], [-69, -12, -52]],
        },
        outputs=[[-71], [-43, -133, -133, -133, -133, -133]],
        solution='Scanl1(lambda u1, u2: Sum(x2), x1)',
    ),
    Task(
        name='synthetic:weight_4_function_7',
        inputs_dict={'x1': [[72, 110, 199], [164, 67, 251], [76, 252, 170]]},
        outputs=[[72, 110], [67, 164], [76, 170]],
        solution='Take(-1, Sort(x1))',
    ),
    Task(
        name='synthetic:weight_4_function_8',
        inputs_dict={
            'x1': [-51, -130, 234],
            'x2': [[3, 0, 3, 4, 0], [0, 3, 1, 3], [5, 3, 3, 5, 5, 3]],
        },
        outputs=[10, 7, 0],
        solution='Sum(Drop(x1, x2))',
    ),
    Task(
        name='synthetic:weight_4_function_9',
        inputs_dict={
            'x1': [[0, 1, 0], [1, 0, 0, 1]],
            'x2': [[0, 1, 1], [1, 1, 0, 0]],
        },
        outputs=[[0], [1, 1]],
        solution='Take(Sum(x1), x2)',
    ),
    Task(
        name='synthetic:weight_5_function_0',
        inputs_dict={
            'x1': [[81, 51, 51, 23], [1], [93, 75, 70, 43, 14]],
            'x2': [[-51, -37], [-97, -33], [-88, -16]],
        },
        outputs=[[-37, -51], [-33], [-16, -88]],
        solution='Take(Head(x1), Reverse(x2))',
    ),
    Task(
        name='synthetic:weight_5_function_1',
        inputs_dict={
            'x1': [
                [21, 3, 3, 33],
                [31, 17, 8, 0],
                [36, 2, 28, 40],
                [23, 0, 25, 20],
            ],
            'x2': [
                [20, 33, 29, 29],
                [22, 32, 5, 50],
                [21, 46, 9, 4],
                [27, 43, 28, 14],
            ],
        },
        outputs=[3, 2, 5, 2],
        solution='IntDivide(Sum(x1), Head(x2))',
    ),
    Task(
        name='synthetic:weight_5_function_2',
        inputs_dict={
            'x1': [
                [-5, -5, -4, -4, -3, -3, -2, -1],
                [-7, -5, -4, -4, -4, -3, -2, -1],
                [-9, -8, -6, -4, -4, -2],
                [-5, -5, -4, -4, -3, -2, -2],
                [-8, -8, -8, -7, -7, -7, -4, -4, -3, -2],
            ]
        },
        outputs=[-4, -3, -2, -4, -2],
        solution='IntDivide(Square(4), Head(x1))',
    ),
    Task(
        name='synthetic:weight_5_function_3',
        inputs_dict={
            'x1': [
                [-9, 0, -4, -7, -6, -3, -5, -10],
                [0, -8, -7, -10, -4, -3, -2],
                [-2, -6, -7, -8, -4, -10, 0, -1, -3],
                [0, -3, -8, -4, -1, -6, -7, -9, -2],
                [0, -10, -1, -9, -7, -8, -5, -3],
            ],
            'x2': [
                [-4, -1, -8, -9, -6, -7, -3, -10],
                [-1, -8, -9, -10, -3, -4, -7],
                [-6, -10, -3, -7, 0, -5, -2, -1, -4],
                [-1, -10, -5, -3, 0, -6, -4, -8, -2],
                [-10, -6, -5, -1, -7, 0, -2, -3],
            ],
        },
        outputs=[
            [-4, 100, 100, 100, 100, 100, 100, 100],
            [-1, 100, 100, 100, 100, 100, 100],
            [-6, 100, 100, 100, 100, 100, 100, 100, 100],
            [-1, 81, 81, 81, 81, 81, 81, 81, 81],
            [-10, 100, 100, 100, 100, 100, 100, 100],
        ],
        solution='Scanl1(lambda u1, u2: Square(Minimum(x1)), x2)',
    ),
    Task(
        name='synthetic:weight_5_function_4',
        inputs_dict={'x1': [4, 1], 'x2': [[3, 1, 0, 1], [2, 5, 4, 0]]},
        outputs=[[3, 0, 0, 0], [2, 2, 2, 2]],
        solution='Scanl1(lambda u1, u2: IntDivide(2, x1), x2)',
    ),
    Task(
        name='synthetic:weight_5_function_5',
        inputs_dict={
            'x1': [-61, 88],
            'x2': [[189, -11], [182, -26]],
            'x3': [-47, -3],
        },
        outputs=[[-11], [182, -26]],
        solution='Drop(IntDivide(x1, x3), x2)',
    ),
    Task(
        name='synthetic:weight_5_function_6',
        inputs_dict={
            'x1': [
                [45, 21],
                [-2, -8, -31, -40],
                [-33, -49],
                [-46, 18, 0, -2],
                [23],
            ],
            'x2': [[32], [-32, 14, 8, -29, -18], [-45], [-41, -42], [46, 7]],
        },
        outputs=[11, 22, 4, -40, -16],
        solution='Subtract(Last(x2), Last(x1))',
    ),
    Task(
        name='synthetic:weight_5_function_7',
        inputs_dict={'x1': [81, -44, -38, -91, 6]},
        outputs=[-16, 15, 14, 27, 3],
        solution='Subtract(4, IntDivide(x1, 4))',
    ),
    Task(
        name='synthetic:weight_5_function_8',
        inputs_dict={
            'x1': [[-45, 22, 21, -1], [-48, -3, -30, -14, 9], [21, 22]]
        },
        outputs=[[-45, 21, -1], [-3, 9], [21]],
        solution='Filter(lambda u1: (lambda v1: IsOdd(v1))(u1), x1)',
    ),
    Task(
        name='synthetic:weight_5_function_9',
        inputs_dict={
            'x1': [6, 9, 0, -1, 2],
            'x2': [
                [38, -36, -9, -6, -25, 10, -18, 11],
                [22, 22, -24, 2, -25, -40, 36, -42, 6, 35],
                [37, -39, -46, 50, -42, 42, -46, -34],
                [41, 48, -2, -37, -36, -19, 34, 9],
                [-12, -2, 40, -18, 44, -33, -18],
            ],
        },
        outputs=[-25, -46, -34, 8, -22],
        solution='Subtract(Last(x2), Square(x1))',
    ),
    Task(
        name='synthetic:weight_6_function_0',
        inputs_dict={
            'x1': [
                [8, 5, 0, 4, 0, 9, 9, 3, 3],
                [9, 1, 8, 3],
                [9, 2, 9, 5, 9, 8, 1],
            ],
            'x2': [
                [-94, -58, -51, -40, 14, 63, 70],
                [-87, -68, -37, 41, 55, 58, 59, 83, 93],
                [-26, -14, -13, -8, 35, 38, 51, 74, 76, 82],
            ],
        },
        outputs=[
            [-94, -3, -3, -3, -3, -3, -3],
            [-87, -3, -3, -3, -3, -3, -3, -3, -3],
            [-26, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        solution='Scanl1(lambda u1, u2: Subtract(0, Last(x1)), x2)',
    ),
    Task(
        name='synthetic:weight_6_function_1',
        inputs_dict={'x1': [-2, 4]},
        outputs=[16, 112],
        solution='Multiply(Square(4), Add(x1, 3))',
    ),
    Task(
        name='synthetic:weight_6_function_2',
        inputs_dict={
            'x1': [
                [1, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            ],
            'x2': [0, 0, 0, 1, 1],
        },
        outputs=[
            [1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, -1, -1, -1, 0, -1, 0, -1],
            [1, 0, 0, 0, -1, -1, -1, 0, 0, 0],
        ],
        solution='Scanl1(lambda u1, u2: (lambda v1: Subtract(v1, x2))(u2), x1)',
    ),
    Task(
        name='synthetic:weight_6_function_3',
        inputs_dict={
            'x1': [[-9, -9, -9, -3], [-3, -6, -7, -7], [-2, -9, -3, -3]],
            'x2': [[-4, -9, -7, -5], [-3, -6, -2, -5], [-8, -7, -7, -4]],
        },
        outputs=[81, 25, 16],
        solution='Square(Min(Head(x1), Last(x2)))',
    ),
    Task(
        name='synthetic:weight_6_function_4',
        inputs_dict={'x1': [[5, 9, 3, 4, 1], [9, 2, 8, 2]]},
        outputs=[[5, 1, 0, 0, 0], [9, 3, 1, 0]],
        solution='Scanl1(lambda u1, u2: (lambda v1: IntDivide(v1, 3))(u1), x1)',
    ),
    Task(
        name='synthetic:weight_6_function_5',
        inputs_dict={
            'x1': [[1, 1, 0], [1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0, 0]]
        },
        outputs=[
            [1, -2, -1],
            [1, -2, -2],
            [1, -1, -2, -2],
            [1, -1, -1, -1, -1],
        ],
        solution='Scanl1(lambda u1, u2: (lambda v1: Subtract(-1, v1))(u2), x1)',
    ),
    Task(
        name='synthetic:weight_6_function_6',
        inputs_dict={
            'x1': [
                [1, 3, 4, 4, 5, 7],
                [1, 2, 2, 3, 6, 6, 7, 7, 9, 9],
                [2, 3, 4, 5, 6, 7],
                [1, 1, 1, 1, 3, 3, 4, 5, 9],
                [1, 1, 1, 2, 3, 6, 7, 8, 8],
            ],
            'x2': [90, 95, 10, 77, 67],
        },
        outputs=[
            [1, 24, 24, 24, 24, 24],
            [1, 52, 52, 52, 52, 52, 52, 52, 52, 52],
            [2, 10, 10, 10, 10, 10],
            [1, 28, 28, 28, 28, 28, 28, 28, 28],
            [1, 37, 37, 37, 37, 37, 37, 37, 37],
        ],
        solution='Scanl1(lambda u1, u2: Min(x2, Sum(x1)), x1)',
    ),
    Task(
        name='synthetic:weight_6_function_7',
        inputs_dict={
            'x1': [-61, 88],
            'x2': [[189, -11], [182, -26]],
            'x3': [-47, -3],
        },
        outputs=[-239, -159],
        solution='Subtract(Min(x1, x3), Sum(x2))',
    ),
    Task(
        name='synthetic:weight_6_function_8',
        inputs_dict={'x1': [[2, 0, 9], [-8, -5, -7]]},
        outputs=[[2, 1, 0], [-8, -4, -2]],
        solution='Scanl1(lambda u1, u2: (lambda v1: IntDivide(v1, 2))(u1), x1)',
    ),
    Task(
        name='synthetic:weight_6_function_9',
        inputs_dict={'x1': [[-231, 79, 1, -5, -150], [172, 105, -57, 174, 0]]},
        outputs=[[-231, 235, -231, 235, -231], [172, -168, 172, -168, 172]],
        solution='Scanl1(lambda u1, u2: (lambda v1: Subtract(4, v1))(u1), x1)',
    ),
    Task(
        name='synthetic:weight_7_function_0',
        inputs_dict={'x1': [-39, -48], 'x2': [26, -2]},
        outputs=[-65, -42],
        solution='Subtract(x1, Min(x2, Multiply(x2, 3)))',
    ),
    Task(
        name='synthetic:weight_7_function_1',
        inputs_dict={'x1': [[-67], [44], [2], [-79]], 'x2': [72, 82, 48, 15]},
        outputs=[[17], [19], [11], [3]],
        solution='Map(lambda u1: IntDivide(Subtract(x2, 3), 4), x1)',
    ),
    Task(
        name='synthetic:weight_7_function_2',
        inputs_dict={
            'x1': [
                [6, -32, -4, 32, 22, -12, 31],
                [0, -47, 18, 18, 39, -5],
                [37, 26, -40, -40],
            ],
            'x2': [[108, 129], [124, 190], [139, 241]],
        },
        outputs=[[5, -33], [-1, -48], [36, 25]],
        solution='ZipWith(lambda u1, u2: (lambda v1: Add(-1, v1))(u1), x1, x2)',
    ),
    Task(
        name='synthetic:weight_7_function_3',
        inputs_dict={'x1': [[0, 1, 1, 0, 0], [1, 1], [0, 1], [1]]},
        outputs=[32, 34, 32, 34],
        solution='Multiply(2, Add(Square(4), Head(x1)))',
    ),
    Task(
        name='synthetic:weight_7_function_4',
        inputs_dict={'x1': [63, 12, 11, 26, 98]},
        outputs=[66, 0, 14, 0, 0],
        solution='If(IsEven(x1), 0, Add(x1, 3))',
    ),
    Task(
        name='synthetic:weight_7_function_5',
        inputs_dict={
            'x1': [31, 63],
            'x2': [[93, 80, 97, 39, 21, 73, 51], [76, 38, 93, 65, 16, 82]],
            'x3': [[0, 1], [1, 0]],
        },
        outputs=[[124, 111], [139, 101]],
        solution='ZipWith(lambda u1, u2: (lambda v1: Add(x1, v1))(u1), x2, x3)',
    ),
    Task(
        name='synthetic:weight_7_function_6',
        inputs_dict={'x1': [-39, -48], 'x2': [26, -2]},
        outputs=[-10, -46],
        solution='Add(x1, Add(3, Max(x2, -1)))',
    ),
    Task(
        name='synthetic:weight_7_function_7',
        inputs_dict={
            'x1': [36, 42, 16, 60],
            'x2': [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            'x3': [
                [5, 9, 10, 5, 0, 10],
                [5, 5],
                [2, 2, 7, 7, 7, 9, 5, 4],
                [10, 5, 10, 9],
            ],
        },
        outputs=[
            [41, 45, 46, 41, 36],
            [47, 47],
            [18, 18, 23, 23, 23, 25, 21],
            [70, 65, 70, 69],
        ],
        solution='ZipWith(lambda u1, u2: (lambda v1: Add(x1, v1))(u2), x2, x3)',
    ),
    Task(
        name='synthetic:weight_7_function_8',
        inputs_dict={
            'x1': [
                [57, 100, 15, 30, 34, 60, 89, 27, 51],
                [17, 30, 73],
                [27, 21, 84, 18, 86, 46, 15],
                [42, 62, 14, 14, 83, 19, 39, 4],
            ],
            'x2': [
                [47, 39, 86, 73, 52, 87, 23, 70, 72],
                [6, 26, 3],
                [70, 26, 65, 10, 35, 90, 98],
                [13, 67, 63, 7, 98, 61, 98, 98],
            ],
            'x3': [
                [-1, -1, -1, -1, -1, -1],
                [-1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
        },
        outputs=[
            [4, 4, 4, 4, 4, 4],
            [0],
            [6, 6, 6, 6, 6, 6, 6, 6],
            [24, 24, 24, 24, 24, 24, 24, 24],
        ],
        solution='Map(lambda u1: IntDivide(Last(x2), Minimum(x1)), x3)',
    ),
    Task(
        name='synthetic:weight_7_function_9',
        inputs_dict={'x1': [8, 6, 9, 9], 'x2': [5, 5, 9, 2]},
        outputs=[10, 7, 10, 14],
        solution='Add(x1, IntDivide(Add(x1, 2), x2))',
    ),
    Task(
        name='synthetic:weight_8_function_0',
        inputs_dict={
            'x1': [
                [1, 1, 1, 1, 0, 0, 0],
                [2, 2, 2, 1, 0, 0, 0, 0, 0],
                [2, 2, 1, 1, 1, 0, 0],
                [2, 2, 1, 1, 1, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 1, 1, 1],
            ]
        },
        outputs=[[1, 4, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        solution=(
            'Take(3, Scanl1(lambda u1, u2: (lambda v1: IntDivide(4, v1))(u1),'
            ' x1))'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_1',
        inputs_dict={
            'x1': [
                [32, 21, 0, 24, 27],
                [36, 30, 36, 29, 21, 45, 18],
                [6, 35, 0, 13, 10, 50, 41, 41, 16, 44],
                [45, 13, 42, 34, 30],
                [9, 21, 23, 45, 46, 50, 30, 20],
            ],
            'x2': [7, 37, 49, 22, 0],
            'x3': [253, 203, 109, 209, 101],
        },
        outputs=[
            [32, 253, 253, 253],
            [45],
            [50],
            [45, 209, 209, 209],
            [9, 101, 101, 101, 101, 101, 101, 101],
        ],
        solution=(
            'Scanl1(lambda u1, u2: x3, Filter(lambda u1: (lambda v1:'
            ' Greater(v1, x2))(u1), x1))'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_2',
        inputs_dict={'x1': [[-238, -195, 77, -187, -200], [182, 12], [-118]]},
        outputs=[[-34, -28, 11, -27, -29], [26, 1], [-17]],
        solution=(
            'Map(lambda u1: (lambda v1: IntDivide(v1, Add(3, 4)))(u1), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_3',
        inputs_dict={
            'x1': [226, 174, 125, 232],
            'x2': [
                [1, 2, 4, 5, 7, 9],
                [3, 4, 6, 9],
                [1, 3, 4, 5, 6, 8, 9],
                [1, 2, 5, 7, 9],
            ],
        },
        outputs=[
            [1, 225, 223, 222, 220, 218],
            [3, 171, 169, 166],
            [1, 123, 122, 121, 120, 118, 117],
            [1, 231, 228, 226, 224],
        ],
        solution=(
            'Scanl1(lambda u1, u2: (lambda v1: Subtract(Add(x1, 1),'
            ' v1))(u2), x2)'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_4',
        inputs_dict={
            'x1': [
                [1, 4, 3, 5, -1],
                [4, -1, 0, 2],
                [3, 0, 4, -1],
                [-1, 5, 2, 1],
                [4, 5, 0],
            ],
            'x2': [3, -1, 1, -1, 2],
            'x3': [
                [-1, 4, 1, 0, 2],
                [3, 1, 2, -1],
                [4, 1, 2, 0],
                [2, 0, 3, 1],
                [2, 1, 3],
            ],
        },
        outputs=[
            [1, 0, 0, 0, -1],
            [4, -1, -1, -1],
            [3, 0, 1, -1],
            [-1, 1, 1, 1],
            [4, 3, 0],
        ],
        solution=(
            'Scanl1(lambda u1, u2: (lambda v1: Min(v1, Access(x2,'
            ' x3)))(u2), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_5',
        inputs_dict={
            'x1': [
                [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1],
                [-1, -1, 0, 0, 1, 1, 1],
                [-1, -1, -1, 0, 0, 0, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, 0, 0, 0, 1, 1],
                [-1, -1, 0, 0, 0, 1, 1, 1, 1, 1],
            ],
            'x2': [
                [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1],
                [-1, -1, -1, 0, 0, 0, 1],
                [-1, -1, -1, -1, -1, 0, 0, 0, 1, 1],
                [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1],
                [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1],
            ],
        },
        outputs=[
            [-2, -2, -2, -2, -2, -2],
            [1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [-3, -3, -3, -3, -3, -3],
            [3, 3, 3, 3, 3, 3],
        ],
        solution=(
            'Map(lambda u1: Sum(x1), Filter(lambda u1: (lambda v1:'
            ' IsOdd(v1))(u1), x2))'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_6',
        inputs_dict={'x1': [8, 6, 6, 5, 9], 'x2': [[4], [6], [4], [9], [8]]},
        outputs=[5, 2, 3, -2, 3],
        solution='Subtract(x1, Max(3, Subtract(Head(x2), 2)))',
    ),
    Task(
        name='synthetic:weight_8_function_7',
        inputs_dict={'x1': [40, 4], 'x2': [[132, 218], [26, 114]]},
        outputs=[[0, 1], [1, 7]],
        solution=(
            'Map(lambda u1: (lambda v1: IntDivide(v1, Multiply(x1,'
            ' 4)))(u1), x2)'
        ),
    ),
    Task(
        name='synthetic:weight_8_function_8',
        inputs_dict={'x1': [37, 39, 31, 49]},
        outputs=[-53, -51, -59, -41],
        solution='Subtract(x1, Add(Square(3), Square(Square(3))))',
    ),
    Task(
        name='synthetic:weight_8_function_9',
        inputs_dict={
            'x1': [
                [0, 9, 3, 10, 6, 9, 7, 5, 10, 4],
                [10, 0],
                [10, 2],
                [0, 2, 10],
            ],
            'x2': [67, 31, 181, 40],
        },
        outputs=[[0, 9, 3, 10, 6, 9, 7, 5, 10, 4], [10, 0], [10, 2], [3, 3, 3]],
        solution='Map(lambda u1: (lambda v1: If(IsEven(x2), 3, v1))(u1), x1)',
    ),
    Task(
        name='synthetic:weight_9_function_0',
        inputs_dict={'x1': [[19, 9, 8], [19, 14, 7], [48, 32, 20, 7]]},
        outputs=[-13, -13, -61],
        solution='Subtract(3, Square(IntDivide(Access(2, Reverse(x1)), 4)))',
    ),
    Task(
        name='synthetic:weight_9_function_1',
        inputs_dict={'x1': [14, 45], 'x2': [[40, 2], [47, 42]]},
        outputs=[[13, 39], [44, 46]],
        solution=(
            'Reverse(Map(lambda u1: (lambda v1: Add(-1, v1))(u1), Scanl1(lambda'
            ' u1, u2: x1, x2)))'
        ),
    ),
    Task(
        name='synthetic:weight_9_function_2',
        inputs_dict={'x1': [34, 165, 79, 52]},
        outputs=[-9, -30, -16, -12],
        solution='Subtract(IntDivide(x1, 3), Add(3, IntDivide(x1, 2)))',
    ),
    Task(
        name='synthetic:weight_9_function_3',
        inputs_dict={
            'x1': [[38, 33, 0, -48], [-17, -46, 28, 12, -41]],
            'x2': [[2, 9], [9, 3, 3, 5, 8]],
        },
        outputs=[[38, -26, -26, -26], [-17, -7, -7, -7, -7]],
        solution=(
            'Scanl1(lambda u1, u2: Subtract(IntDivide(Last(x1), Head(x2)),'
            ' 2), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_9_function_4',
        inputs_dict={
            'x1': [
                [48],
                [-3, 14, 19, 45, 9],
                [8, 9, 15, 17, 4],
                [15, 38],
                [33, 20, 26],
            ],
            'x2': [29, 8, 42, 30, 30],
        },
        outputs=[48, 2, 125, 42, 87],
        solution=(
            'Sum(Take(Minimum(x1), Scanl1(lambda u1, u2: Subtract(x2, 3), x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_9_function_5',
        inputs_dict={
            'x1': [
                [70, 56, 91, 17, 69, 28],
                [80, 44, 64, 28, 19, 47],
                [64, 89, 24, 47, 83, 42],
            ]
        },
        outputs=[
            [13, -3, -3, -3, -3, -3],
            [15, -3, -3, -3, -3, -3],
            [20, -3, -3, -3, -3, -3],
        ],
        solution=(
            'Map(lambda u1: (lambda v1: Subtract(v1, 4))(u1), Scanl1(lambda u1,'
            ' u2: 1, Sort(x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_9_function_6',
        inputs_dict={
            'x1': [[0], [2, 8], [8, 6, 2], [3, 2], [9]],
            'x2': [
                [20, -40, -47, 50, 13, 25, 43, -48, 42],
                [11, 43, -11, -6, -13, -13, -15],
                [-25, 46, 4, 43, -43, 0],
                [47, 33, -4, 25, 46, -46, -14, -10, -16, 4],
                [-2, 43, 50, -37, -14, -38, -46, -26],
            ],
        },
        outputs=[[6], [86, 86], [92, 92, 92], [66, 66], [86]],
        solution='Map(lambda u1: Multiply(2, Max(3, Access(1, x2))), x1)',
    ),
    Task(
        name='synthetic:weight_9_function_7',
        inputs_dict={
            'x1': [[-9, -2, -10, -6, 0, -10, -6, 3, 1], [-1, -5, 8, 5]]
        },
        outputs=[[0, 0, 0, 0, 0, 0, 0, 3, 1], [0, 0, 4, 4]],
        solution=(
            'Map(lambda u1: (lambda v1: Min(4, (lambda v1: Max(0,'
            ' v1))(v1)))(u1), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_9_function_8',
        inputs_dict={
            'x1': [[7], [10, 8, 1, 3, 6], [6, 6], [5, 8, 10]],
            'x2': [[-37, 49], [25, 156], [-106, 83], [-179, -100]],
            'x3': [140, 26, -226, -196],
        },
        outputs=[[143], [29, 156], [-106, 83], [-179, -100]],
        solution=(
            'ZipWith(lambda u1, u2: (lambda v1: Max(v1, Add(x3, 3)))(u2),'
            ' x1, x2)'
        ),
    ),
    Task(
        name='synthetic:weight_9_function_9',
        inputs_dict={'x1': [[90], [26]]},
        outputs=[[67], [19]],
        solution=(
            'Map(lambda u1: (lambda v1: IntDivide((lambda v1: Multiply(3,'
            ' v1))(v1), 4))(u1), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_0',
        inputs_dict={
            'x1': [178, 95, -7, 63],
            'x2': [
                [28, 32, 37],
                [-42, -34, -13, 42, 47],
                [-42, -21, 10, 35],
                [5, 25],
            ],
            'x3': [
                [3, 126, 134, 140, 171, 178, 228],
                [2, 13, 35, 93, 185, 220],
                [45, 62, 81, 89],
                [60, 116, 149, 214],
            ],
        },
        outputs=[71, 47, 32, 47],
        solution='IntDivide(Add(Last(x2), Max(x1, Access(1, x3))), 3)',
    ),
    Task(
        name='synthetic:weight_10_function_1',
        inputs_dict={'x1': [[-1, -1, 4], [2, 4, 8], [-1, 6, 9]]},
        outputs=[[-1, 1, 2], [0, 1, 2], [-1, 0, 0]],
        solution=(
            'Sort(Scanl1(lambda u1, u2: (lambda v1: IntDivide(4, (lambda v1:'
            ' Max(2, v1))(v1)))(u2), x1))'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_2',
        inputs_dict={'x1': [34, 165, 79, 52]},
        outputs=[-5, -1, 0, 11],
        solution=(
            'Subtract(IntDivide(x1, Subtract(x1, Multiply(3, Square(4)))), 2)'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_3',
        inputs_dict={'x1': [[2, 0, 9], [-8, -5, -7]]},
        outputs=[[4, 100, 100], [64, 0, 0]],
        solution=(
            'Map(lambda u1: (lambda v1: Square(v1))(u1), Scanl1(lambda u1, u2:'
            ' Sum(Scanl1(lambda u1, u2: 4, x1)), x1))'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_4',
        inputs_dict={
            'x1': [
                [0, 1, 0, 1, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 1],
                [1, 0, 1, 0, 1],
            ],
            'x2': [
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 1, 0],
                [0, 0, 1, 1, 0, 1, 0],
                [0, 1, 0, 1, 1],
            ],
            'x3': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        },
        outputs=[[4, 4], [9, 9], [1, 1], [9, 9], [9, 9]],
        solution=(
            'Map(lambda u1: Square(Count(lambda u1: (lambda v1: Equal(v1,'
            ' Last(x1)))(u1), x2)), x3)'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_5',
        inputs_dict={
            'x1': [[0, 1, 0], [1, 0, 0, 1]],
            'x2': [[0, 1, 1], [1, 1, 0, 0]],
        },
        outputs=[[-2, 1, 1], [-2, 0, 0]],
        solution=(
            'Take(3, Scanl1(lambda u1, u2: Last(x2), Map(lambda u1: Add(-1,'
            ' -1), x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_6',
        inputs_dict={'x1': [[1], [-4], [-1]]},
        outputs=[[1], [-254], [1]],
        solution=(
            'Map(lambda u1: (lambda v1: Subtract(2, (lambda v1: Square((lambda'
            ' v1: Square(v1))(v1)))(v1)))(u1), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_7',
        inputs_dict={
            'x1': [
                [1, 4, 3, 5, -1],
                [4, -1, 0, 2],
                [3, 0, 4, -1],
                [-1, 5, 2, 1],
                [4, 5, 0],
            ],
            'x2': [3, -1, 1, -1, 2],
            'x3': [
                [-1, 4, 1, 0, 2],
                [3, 1, 2, -1],
                [4, 1, 2, 0],
                [2, 0, 3, 1],
                [2, 1, 3],
            ],
        },
        outputs=[9, -2, 2, -7, 2],
        solution=(
            'Add(x2, Sum(Map(lambda u1: (lambda v1: IntDivide(v1,'
            ' Head(x1)))(u1), x3)))'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_8',
        inputs_dict={
            'x1': [95, -121, 170, 179],
            'x2': [[1, 1], [1, 1], [1, 1], [1, 1]],
            'x3': [
                [-159, -150, -87, -180],
                [-182],
                [-188, -200, -70, -93],
                [-7, -186, -119, -169, -193, -187, -166, -135, -80],
            ],
        },
        outputs=[[1, -1], [-121, -1], [1, -1], [1, -26]],
        solution=(
            'Scanl1(lambda u1, u2: (lambda v1: IntDivide(v1, Head(x3)))(u2),'
            ' Sort(Scanl1(lambda u1, u2: x1, x2)))'
        ),
    ),
    Task(
        name='synthetic:weight_10_function_9',
        inputs_dict={
            'x1': [[-10, -6, -5, -4, -2, 6], [-10, -9, -7, -7, -5, -2]]
        },
        outputs=[[10, 0, 0, 0, 0], [13, 0, 0, 0, 0]],
        solution=(
            'Map(lambda u1: (lambda v1: Subtract(4, v1))(u1), Scanl1(lambda u1,'
            ' u2: 4, Drop(1, x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_0',
        inputs_dict={
            'x1': [3, 3, 5],
            'x2': [
                [2, 3, 3, 5, 8, 8, 9],
                [1, 1, 3, 3, 5, 5, 7, 8, 8, 8],
                [2, 3, 4, 5, 6, 6, 6, 8, 9],
            ],
        },
        outputs=[-22, -60, -78],
        solution=(
            'Add(3, IntDivide(Sum(Scanl1(lambda u1, u2: (lambda v1:'
            ' Subtract(v1, x1))(u1), x2)), 2))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_1',
        inputs_dict={'x1': [19, 10, 16, 7], 'x2': [36, 43, 13, 21]},
        outputs=[1, -46, 0, -21],
        solution=(
            'If(IsEven(x2), 1, Subtract(x1, Multiply(4, IntDivide(x2, 3))))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_2',
        inputs_dict={
            'x1': [
                [-1, 0],
                [-1, -1, 1, 0, 0],
                [1, 0, 0, 1, 1, 0, -1, 1],
                [-1, 1, 1, 0],
                [-1, 0, 0, -1, -1, 0],
            ]
        },
        outputs=[0, -16, 2, 0, -48],
        solution=(
            'Min(2, Sum(Drop(4, Scanl1(lambda u1, u2: (lambda v1: Add(v1,'
            ' v1))(u1), x1))))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_3',
        inputs_dict={
            'x1': [[1, 1], [2, 2], [1, 0], [0, 1]],
            'x2': [2, 0, 2, 0],
        },
        outputs=[-36, 0, -9, -1],
        solution=(
            'Subtract(0, Square(Add(x2, Square(Count(lambda u1: (lambda v1:'
            ' IsOdd(v1))(u1), x1)))))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_4',
        inputs_dict={'x1': [2, 0, 4]},
        outputs=[-20, 16, -164],
        solution=(
            'Multiply(4, Subtract(4, Multiply(Square(3), IntDivide(Square(x1),'
            ' 3))))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_5',
        inputs_dict={
            'x1': [
                [193, 228, 228, 215, 143],
                [229, 116, 186],
                [232, 243, 244, 186],
            ],
            'x2': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        },
        outputs=[[51], [], [51]],
        solution=(
            'Drop(3, ZipWith(lambda u1, u2: Add(2, Square(Add(3, 4))), x1, x2))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_6',
        inputs_dict={
            'x1': [
                [-117],
                [-166],
                [-99, -202, -90, -14],
                [-221, -222, -241, -193, -213],
                [-248],
            ]
        },
        outputs=[[], [-166], [-13, -25, -50, -99], [], [-248]],
        solution=(
            'Reverse(Filter(lambda u1: IsEven(Last(x1)), Scanl1(lambda u1, u2:'
            ' (lambda v1: IntDivide(v1, 2))(u1), x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_7',
        inputs_dict={
            'x1': [[-20, 63], [0, -42], [93, 45], [-41, -94], [-9, 35]]
        },
        outputs=[63, -3, 138, -3, 33],
        solution='Multiply(3, IntDivide(Max(-1, Max(Last(x1), Sum(x1))), 3))',
    ),
    Task(
        name='synthetic:weight_11_function_8',
        inputs_dict={'x1': [[8], [0, 2]]},
        outputs=[-13, 196],
        solution=(
            'IntDivide(Square(Subtract(Square(4), Last(x1))), Subtract(3,'
            ' Last(x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_11_function_9',
        inputs_dict={
            'x1': [
                [-95],
                [-99, -47, -202, -131, -161, 180, 216, -193],
                [184, 117, -154, 4, 41, 15],
                [246, 129, 14, 78, 256, -95, -74],
            ]
        },
        outputs=[[-95], [-99, 19], [184, 187], [246, 249]],
        solution=(
            'Take(2, Scanl1(lambda u1, u2: Add(3, Max(Square(4), Head(x1))),'
            ' x1))'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_0',
        inputs_dict={'x1': [[204], [31]]},
        outputs=[[212], [4]],
        solution=(
            'Map(lambda u1: (lambda v1: IntDivide((lambda v1: Square(v1))(v1),'
            ' Square(Subtract(2, Square(4)))))(u1), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_1',
        inputs_dict={'x1': [[0], [0, 1]]},
        outputs=[[144], [72, 144]],
        solution=(
            'Reverse(Scanl1(lambda u1, u2: (lambda v1: IntDivide(v1, 2))(u1),'
            ' Map(lambda u1: Square(Multiply(3, 4)), x1)))'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_2',
        inputs_dict={'x1': [1, 0]},
        outputs=[-177, -118],
        solution=(
            'Multiply(Add(x1, 2), Add(1, Subtract(4, Multiply(4, Square(4)))))'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_3',
        inputs_dict={'x1': [96, 79]},
        outputs=[-11, 156],
        solution=(
            'Subtract(If(IsOdd(x1), Add(x1, x1), Subtract(0, Square(3))), 2)'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_4',
        inputs_dict={'x1': [158, 232], 'x2': [0, 0]},
        outputs=[-4, 237],
        solution=(
            'Subtract(If(IsEven(IntDivide(x1, 3)), x2, Add(x1, Square(3))), 4)'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_5',
        inputs_dict={'x1': [115, 177], 'x2': [0, 0]},
        outputs=[-16, -196],
        solution=(
            'Subtract(x2, Square(Add(Square(4), IntDivide(x1, Subtract(3,'
            ' Square(3))))))'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_6',
        inputs_dict={'x1': [158, 232], 'x2': [0, 0]},
        outputs=[16, -218],
        solution=(
            'Subtract(Square(4), If(IsEven(IntDivide(x1, 3)), x2, Add(x1, 2)))'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_7',
        inputs_dict={'x1': [115, 177], 'x2': [0, 0]},
        outputs=[-138, -196],
        solution=(
            'Subtract(x2, Add(x1, IntDivide(x1, Subtract(IntDivide(x1,'
            ' Square(4)), 2))))'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_8',
        inputs_dict={'x1': [[1], [0]]},
        outputs=[[-225], [-256]],
        solution=(
            'Map(lambda u1: (lambda v1: Subtract(0, (lambda v1: Square((lambda'
            ' v1: Subtract(v1, Square(4)))(v1)))(v1)))(u1), x1)'
        ),
    ),
    Task(
        name='synthetic:weight_12_function_9',
        inputs_dict={'x1': [67, 33, 99]},
        outputs=[137, 115, 201],
        solution=(
            'Add(3, Multiply(2, Max(x1, Multiply(4, Subtract(Square(4), 2)))))'
        ),
    ),
]
# pylint: enable=line-too-long

# Some task names were changed as we became aware of solutions with smaller
# weight than our original handwritten solutions. The old names might persist in
# logs for approaches run before the rename. These changes don't affect any I/O
# examples or the function or argument names.
RENAMING_MAP = {
    'map:rank': 'count:rank',
    'multi:even_or_positive': 'count:even_or_positive',
    'multi:deepcoder_3': 'scanl1:deepcoder_3',
}


def get_task(task_name):
  if task_name in RENAMING_MAP:
    task_name = RENAMING_MAP[task_name]

  search_in = (SYNTHETIC_TASKS if 'synthetic:' in task_name
               else HANDWRITTEN_TASKS)
  matches = [task for task in search_in if task.name == task_name]
  if len(matches) != 1:
    raise ValueError(f'Task name {task_name} matched {len(matches)} tasks')
  return matches[0]


def print_stats(tasks):
  """Prints some stats about the tasks."""
  op_counts = collections.Counter()
  num_ops = collections.Counter()
  weights = collections.Counter()
  prefixes = collections.Counter()
  for task in tasks:
    ops_used = re.findall(r'\b([A-Z]\w+)\(', task.solution)
    op_counts.update(ops_used)
    num_ops[len(ops_used)] += 1
    weights[solution_weight.solution_weight(task.solution)] += 1
    prefixes[task.name.split(':')[0]] += 1
  print(f'There are {len(tasks)} tasks.')
  print(f'Operation usage counts: {op_counts}')
  print(f'Number of operations used: {dict(sorted(num_ops.items()))}')
  print(f'Solution weights: {dict(sorted(weights.items()))}')
  print(f'Name prefix counts: {prefixes}')


if __name__ == '__main__':
  print('Handwritten tasks:')
  print_stats(HANDWRITTEN_TASKS)
  print('\nSynthetic tasks:')
  print_stats(SYNTHETIC_TASKS)

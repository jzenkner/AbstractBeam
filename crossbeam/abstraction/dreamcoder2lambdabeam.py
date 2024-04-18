import re
import torch
from crossbeam.abstraction.utils import get_function_dict
from crossbeam.dsl.deepcoder_operations import DeepCoderOperation
from crossbeam.dsl.deepcoder_utils import run_program
from collections import OrderedDict
import functools

def find_matching_parenthesis_index(tokens, open_parenthesis_index):
    stack = []
    for i, token in enumerate(tokens):
        if token == '(':
            stack.append(i)
        elif token == ')' and stack:
            if stack[-1] == open_parenthesis_index:
                return i
            else:
                stack.pop()

    return None  # No matching closing parenthesis found


def count_arity(tokens, index, base_function_dict):
    end_idx = find_matching_parenthesis_index(tokens, index) + 1
    tokens = tokens[index:end_idx]
    arity = 0
    parantheses_counter = -1
    for i, token in enumerate(tokens):
        if token == '(':
            if tokens[i+1] == "lambda" and parantheses_counter == 0:
                arity += 1
            parantheses_counter += 1
        elif token == ')':
            parantheses_counter -= 1
        elif parantheses_counter == 0 and (token == "[" or (token in base_function_dict.keys() or token.isdigit() or (token[0] == "-" and token[1:].isdigit()) or (token[0] in ["v", "u", "p", "q", "r", "s", "t", "w", "y", "z", "f", "x", "$", "#"] and token[1:].isdigit() and tokens[i-1] != "lambda"))):
            arity += 1
    
    return arity, end_idx

def find_matching_parenthesis_index(tokens, open_parenthesis_index):
    stack = []
    for i, token in enumerate(tokens):
        if token == '(':
            stack.append(i)
        elif token == ')' and stack:
            if stack[-1] == open_parenthesis_index:
                return i
            else:
                stack.pop()

    return None  # No matching closing parenthesis found

def find_max_global_var(input_string, pattern=r'x(\d+)'):
    matches = re.findall(pattern, input_string)
    
    if not matches:
        return -1
    max_number = max(map(int, matches))
    return max_number


def add_missing_args(result_string, base_function_dict):
    # find the max number in operation which is preceded by a x in the operation
    max_global_var = find_max_global_var(result_string)
    var_index = max_global_var + 1
    if var_index == 0:
        var_index = 1
    tokens = [token for token in re.findall(r'-?\b\w+\b|\#\d||\$\d|[():+-\[\]]|,|', result_string) if token.strip()]
    for i, token in enumerate(tokens):
        if token in base_function_dict.keys():
            arity, end_idx = count_arity(tokens, i+1, base_function_dict)
            while arity != base_function_dict[token]:
                tokens.insert(end_idx-1, f",x{var_index}")
                arity += 1
                var_index += 1
    return "".join(tokens)


def calculate_bound_variables_dict(result_string, base_function_dict, higher_order_functions = {"Map": [1,0], "Fold": [2,0,0]}):
    bound_variables_dict = OrderedDict()
    tokens = [token for token in re.findall(r'-?\b\w+\b|\#\d||\$\d|[():+-\[\]]|,|', result_string) if token.strip()]
    arguments_counter = OrderedDict()
    for i, token in enumerate(tokens):
        if token in base_function_dict:
            if len(arguments_counter) > 0:
                arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
            if len(arguments_counter) > 0 and arguments_counter[list(arguments_counter.items())[-1][0]] == 0:
                arguments_counter.popitem()
            arguments_counter[token] = base_function_dict[token]

        elif token[0] == "x":
            if len(arguments_counter) > 0:  
                func = list(arguments_counter.items())[-1][0]
            else:
                func = None
            if token[1:].isdigit() and tokens[i+1] == "(":
                arity, _ = count_arity(tokens, i+1, base_function_dict)
                if token in bound_variables_dict:
                    new_arity = min(arity, bound_variables_dict[token][0])
                    bound_variables_dict[token][0] = new_arity
                    if func is not None and "x" not in func:
                        bound_variables_dict[token][1].append(func)

                else:
                    if func is not None and "x" not in func:
                        bound_variables_dict[token] = [arity, [func]]
                    else:
                        bound_variables_dict[token] = [arity, [None]]
                if len(arguments_counter) > 0:
                    arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
            
                if len(arguments_counter) > 0 and arguments_counter[list(arguments_counter.items())[-1][0]] == 0:
                    arguments_counter.popitem()
                arguments_counter[token] = arity
                continue
            elif token[1:].isdigit():
                # get key of last elem in ordered arguments_counter dict
                func = list(arguments_counter.items())[-1][0]
                if "x" in func:
                    if token in bound_variables_dict:
                        bound_variables_dict[token][1].append(None)
                    else:
                        bound_variables_dict[token] = [0, [None]]
                else:
                    if token in bound_variables_dict:
                        if list(arguments_counter.items())[-1][0] in higher_order_functions.keys() and higher_order_functions[list(arguments_counter.items())[-1][0]][-int(list(arguments_counter.items())[-1][1])] > 0:
                            new_arity = min(higher_order_functions[list(arguments_counter.items())[-1][0]][-int(list(arguments_counter.items())[-1][1])], bound_variables_dict[token][0])
                        else:
                            new_arity = min(0, bound_variables_dict[token][0])
                        bound_variables_dict[token][0] = new_arity
                        bound_variables_dict[token][1].append(func)
                    else:
                        if list(arguments_counter.items())[-1][0] in higher_order_functions.keys() and higher_order_functions[list(arguments_counter.items())[-1][0]][-int(list(arguments_counter.items())[-1][1])] > 0:
                            bound_variables_dict[token] = [higher_order_functions[list(arguments_counter.items())[-1][0]][-int(list(arguments_counter.items())[-1][1])], [func]]
                        else:
                            bound_variables_dict[token] = [0, [func]]
                
            if len(arguments_counter) > 0:
                arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
            
            if len(arguments_counter) > 0 and arguments_counter[list(arguments_counter.items())[-1][0]] == 0:
                arguments_counter.popitem()
        
        elif token.isdigit() or token == "[" or (token[0] == "-" and token[1:].isdigit()) or (token[0].isalpha() and token[1:].isdigit() and tokens[i-1] != "lambda"):
            if len(arguments_counter) > 0:
                arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
            if len(arguments_counter) > 0 and arguments_counter[list(arguments_counter.items())[-1][0]] == 0:
                arguments_counter.popitem()

    return bound_variables_dict


def insert_commas(result_string):
    tokens = [token for token in re.findall(r'-?\b\w+\b|\$\d|[():+*-]', result_string) if token.strip()]
    for i, token in enumerate(tokens):
        if (token[0] in ["x", "v", "u", "p", "q", "r", "s", "t", "w", "y", "z"] and token[1:].isdigit()) or token == ")" and tokens[i+1] not in [")", ",", "("]:
            tokens.insert(i+1, ", ")


def remove_lambdas(result_string, base_function_dict):
    tokens = [token for token in re.findall(r'-?\b\w+\b|\#\d||\$\d|[():+-\[\]]|,|', result_string) if token.strip()]
    starting = 0
    elem_parenthesis = 0
    for i, token in enumerate(tokens):
       if token == "(":
           elem_parenthesis += 1
       if token in base_function_dict or (token[0] == "x" and token[1:].isdigit() and tokens[i+1] == "("):
            starting = i
            break
    
    if elem_parenthesis == 0:
        return "".join(tokens[starting:])
    else:
        return "".join(tokens[starting:-elem_parenthesis])

def correct_double_lam(result_string):
    new_tokens = []
    skip = False
    tokens = [token for token in re.findall(r'-?\b\w+\b|\#\d||\$\d|[():+-\[\]]|,|', result_string) if token.strip()]
    for i, token in enumerate(tokens):
        if token == ":" and tokens[i+1] == "(" and tokens[i+2] == "lambda":
            a = f"{tokens[i+3]},"
            # insert a infront of last elem in new_tokens
            new_tokens.insert(-1, a)
            new_tokens.append(":")
            skip = True
        elif (token == ":" or token == "lambda" or token == "(" or (token[1:].isdigit() and token[0].isalpha() and not tokens[i+1] == "(")) and skip:
            if token == ":":
                skip = False
            if token == "(":
                new_tokens.append("(")
            continue
        else:
            skip = False
            new_tokens.append(token)

    new_program = "".join(new_tokens)
    new_program = new_program.replace("lambda", "lambda ")
    new_program = new_program.replace(":", ": ")
    new_program = new_program.replace(",", ", ")
    return new_program

def reduce_variables(expr, missing_var):
    # Regular expression pattern to match variables
    pattern = r'\b(x)(\d+)\b'

    def replace_variable(match):
        prefix = match.group(1)
        number = int(match.group(2))
        if number > missing_var:
            return f"{prefix}{number - 1}"
        else:
            return f"{prefix}{number}"


    # Replace variables in the expression
    incremented_expr = re.sub(pattern, replace_variable, expr)
    return incremented_expr

def check_variable_ordering(expr):

    # Check if expression contains variables
    variabeles = set(re.findall(r'\b(x)(\d+)\b', expr))
    if len(variabeles) == 0:
        return expr

    # concatenate tuples
    variabeles = ["".join(var) for var in variabeles]
    max_var = max([int(var[1]) for var in variabeles])
    missing_vars = []
    for i in range(1, max_var+1):
        if str(i) not in [var[1] for var in variabeles]:
            missing_vars.append(i)

    # reverse the list
    missing_vars = missing_vars[::-1]
    for missing_var in missing_vars:
        expr = reduce_variables(expr, missing_var)
    else:
        return expr


def remove_outer_x_functions(result_string):
    if result_string[0] == "x" and result_string[1].isdigit():

        return result_string[3:-1]
    return result_string


def remove_unnecessary_x_functions(result_string):
    pattern = r'x\d+\(x\d+\)'
    matches = re.findall(pattern, result_string)
    matches = list(set(matches))
    for match in matches:
        no_replacement = False
        used_vars = match.split("(")
        incomplete_result_string = result_string.replace(match, "")
        for var in used_vars:
            if var in incomplete_result_string:
                no_replacement = True
                break
        if no_replacement:
            continue
        result_string = result_string.replace(match, match.split("(")[0])
    
    return result_string



def parse_abstraction(abstraction, base_function_dict, higher_order_functions = {"Map": [1,0], "Fold": [2,0,0]}):
    tokens = [token for token in re.findall(r'-?\b\w+\b|\#\d||\$\d|[():+-]', abstraction) if token.strip()]
    local_vars = ["v", "u", "p", "q", "r", "s", "t", "w", "y", "z"]
    local_var_counter = -1
    open_parentheses_count = 0
    local_var_parentheses = []
    bound_variables_dict = OrderedDict()
    max_hashtag_var = find_max_global_var(abstraction, pattern='#(\d+)')
    result_string = ""
    arguments_counter = OrderedDict()
    if tokens[0] == "(" and tokens[1] == "lam":
        tokens = tokens[2:-1]
    for i, token in enumerate(tokens):
        if token == "(":
            open_parentheses_count += 1

            result_string += token

        elif token == ")":
            if len(local_var_parentheses) > 0 and local_var_parentheses[-1] == open_parentheses_count:
                local_var_parentheses.pop()
                local_var_counter -= 1
            open_parentheses_count -= 1
            
            result_string += token
            if i != len(tokens)-1 and tokens[i+1] != ")" and tokens[i+1] != "," and tokens[i-1] != "(":
                result_string += ","

        elif token in base_function_dict.keys():
            result_string = result_string[:-1] + token + result_string[-1]
            if len(arguments_counter) > 0:
                arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
            if len(arguments_counter) > 0 and arguments_counter[list(arguments_counter.items())[-1][0]] == 0:
                arguments_counter.popitem()
            arguments_counter[token] = base_function_dict[token]

        elif token == "lam" or token == "lambda":
            local_var_counter += 1
            local_var_parentheses.append(open_parentheses_count)

            result_string += f"lambda {local_vars[local_var_counter]}1: "
            
        # replace variables
        elif (token[0] == "$" or token[0] == "#") and token[1:].isdigit():

            # var is a function
            if tokens[i-1] == "(":
                func_name = "x" + str(int(token[1:]) + 1) if token[0] == "#" else "x" + str(int(token[1:]) + 2 + max_hashtag_var)
                result_string = result_string[:-1] + func_name + result_string[-1]
                if len(arguments_counter) > 0:
                    arguments_counter[list(arguments_counter.items())[-1][0]] -= 1

                # add arguments to arguments_counter
                arguments_counter[func_name] = count_arity(tokens, i-1, base_function_dict)[0] - 1
                

            elif token[0] == "#":
               result_string += "x" + str(int(token[1:]) + 1)
               
            elif token[0] == "$":
                if int(token[1:]) <= local_var_counter:
                    result_string += str(local_vars[local_var_counter - int(token[1:])]) + "1"
                else:
                    result_string += "x" + str(int(token[1:]) - local_var_counter - 1 + max_hashtag_var + 2)
            
            if i != len(tokens)-1 and tokens[i+1] != ")" and tokens[i+1] != "," and tokens[i-1] != "(":
                result_string += ","
            
            if len(arguments_counter) > 0:
                arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
        elif token.isdigit() or (token[0] == "-" and token[1:].isdigit()) or token == "empty":
            if token == "empty":
                result_string += "[]"
            else:
                result_string += token
            if len(arguments_counter) > 0:
                arguments_counter[list(arguments_counter.items())[-1][0]] -= 1
            
            if i != len(tokens)-1 and tokens[i+1] != ")" and tokens[i+1] != "," and tokens[i-1] != "(":
                result_string += ","
    
    
    # add missing arguments     
    result_string = add_missing_args(result_string, base_function_dict)
    # remove precedings lambdas 
    result_string = remove_lambdas(result_string, base_function_dict)
    # add necessary spaces 
    result_string = result_string.replace("lambda", "lambda ")
    result_string = result_string.replace(":", ": ")
    result_string = result_string.replace(",", ", ")

    # remove unnecessary x functions
    result_string = remove_unnecessary_x_functions(result_string)

    # check variable ordering
    result_string = check_variable_ordering(result_string)

    # calculate bound variables
    bound_variables_dict = calculate_bound_variables_dict(result_string, base_function_dict, higher_order_functions)
    # correct double lambdas
    result_string = correct_double_lam(result_string)
    

    return result_string, bound_variables_dict



        
class Invented(DeepCoderOperation):
  def __init__(self, arity, program, invention, dc_program = None, bound_variables = [], dc_prims = True, inventions = None):
    super(Invented, self).__init__(arity, num_bound_variables = bound_variables, inv_name = invention.name)
    self.program = program
    self.invention = invention
    self.arity = arity
    self.dc_program = dc_program
    self.dc_prims = dc_prims
    self.inventions = inventions

  def apply_single(self, raw_args):
    arguments = raw_args
    if len(arguments) != self.arity:
      return None
    input_dict = {f"x{i+1}": [arg] for i, arg in enumerate(arguments)}
    if not self.dc_prims:
        self.dc_prims = True
    try:
        output = run_program(self.program, input_dict, self.dc_prims, self.inventions)[0]
    except AttributeError as e:
        output = run_program(self.program, input_dict, self.dc_prims, [])[0]
    if output is None or(type(output) == list and None in output):
        return None
    else:
        return output


def contains_only_one_operation(main_string, base_function_dict):
    operations = list(base_function_dict.keys()) # upper case

    # Initialize a dictionary to store the counts
    substring_counts = {}

    # Iterate through each substring in the list
    for op in operations:
        # Count occurrences of the substring in the main string
        count = main_string.count(op + "(")

        # Store the count in the dictionary
        substring_counts[op] = count
    # Check if the total count of matching substrings is 2 or more
    total_count = sum(substring_counts.values())

    # regex exrpression for xi( where i can be any positive integer

    return substring_counts, total_count < 2


def contains_no_constants(main_string):
    for i in range(-300, 300):
        if f" {i})" in main_string or f" {i}, " in main_string or f"({i} " in main_string or f"({i}, " in main_string or f" -{i})" in main_string or f" -{i}, " in main_string or f"(-{i} " in main_string or f"(-{i}, " in main_string or "[]" in main_string:
            return False
        
    return True


def rewrite_abstractions(abstractions, base_function_dict):
    for abs in abstractions:
        while "fn_" in abs.body:
            pattern = r'\bfn_\d+\b'
            matches = re.finditer(pattern, abs.body)
            result = [(match.group(), match.start()) for match in matches]
            name, start_index = result[0]


            if name in base_function_dict.keys():
                abs.body = re.sub(r'\b' + re.escape(name) + r'(?!\d)', "y" + name[1:], abs.body)
                continue
            
            
            arguments = []
            no_arguemnts = False
            if abs.body[start_index-1] == "(":
                end_idx = find_matching_parenthesis_index(abs.body[start_index-1:], 0)
                i = start_index
                while i < start_index+end_idx:
                    
                    if abs.body[i] == "(":
                        arguments_end_idx = find_matching_parenthesis_index(abs.body[i:], 0)
                        arguments.append(abs.body[i:i+arguments_end_idx+1])
                        i = i + arguments_end_idx
                    elif abs.body[i] in ["#", "$"]:
                        arguments.append(abs.body[i:i+2]) # Assuming not more arguemnts than 9
                        i = i+1
                    elif (abs.body[i].isdigit() or (abs.body[i] == "-" and abs.body[i+1].isdigit())) and abs.body[i-1] in [" ", "("]:
                        # get whole number
                        number = ""
                        while abs.body[i].isdigit() or abs.body[i] == "-":
                            number += abs.body[i]
                            i += 1
                        arguments.append(number)
                    
                    # is a character
                    elif abs.body[i].isalpha():
                        # get whole word
                        word = ""
                        while abs.body[i].isalpha():
                            word += abs.body[i]
                            i += 1
                        
                        if word in base_function_dict.keys() or word == "empty":
                            arguments.append(word)
                    else:
                        i += 1
            else:
                # no arguments and no parenthesis
                no_arguemnts = True

            # so that arguments are not replaced twice
            for i in range(len(arguments)):
                arguments[i] = arguments[i].replace("#", "!")
            
            for abs2 in abstractions:
                if abs2.name == name:
                    if no_arguemnts:
                        abs.body = abs.body.replace(name, abs2.body)
                        break
                    
                    if not "#" in abs2.body and not no_arguemnts:
                        abs.body = abs.body.replace(name, abs2.body[1:-1])
                        continue
                    abs.body = abs.body[:start_index-1] + name + abs.body[end_idx+1+max(0, len(abs.body[:start_index]) - 1):]
                    inner_func = abs2.body
                    for i, char in enumerate(abs2.body):
                        if char == "#":
                            if len(arguments) - 1 < int(abs2.body[i+1]):
                                continue
                            elif abs2.body[i-1] == "(" and arguments[int(abs2.body[i+1])][0] == "(":
                                inner_func = inner_func.replace(abs2.body[i:i+2], arguments[int(abs2.body[i+1])][1:-1])
                            else:
                                inner_func = inner_func.replace(abs2.body[i:i+2], arguments[int(abs2.body[i+1])])
            
                    abs.body = abs.body.replace(name, inner_func)
                    break
            abs.body = abs.body.replace("!", "#")

        abs.body = abs.body.replace("yn_", "fn_")

    return abstractions

def get_outer_op(program, base_function_dict):
    tokens = [token for token in re.findall(r'-?\b\w+\b|\#\d||\$\d|[():+-]|,|', program) if token.strip()]
    for token in tokens:
        if token in base_function_dict.keys():
            return token


def build_inventions(new_inventions, old_inventions, higher_order_functions, base_function_dict, model=None, optimizer=None, device=None, lr=None, initialization_method = "top", dc_abstractions = [], pruning = False, domain=None, max_invention = 15, inv_namespace = None, abstraction_refinement=None):
    # rewrite new abstractions
    dc_counter = 0

    new_inventions = rewrite_abstractions(new_inventions, base_function_dict)
    added = False
    dc_prims = "Mod" in base_function_dict.keys()
    skipped_inventions = {}
    for invention in new_inventions:
        if dc_counter > max_invention:
            break
        print("abstraction:" , invention.body)
        try:
            lambdabeam_program, bound_variables_dict = parse_abstraction(invention.body, base_function_dict, higher_order_functions)
        except Exception as e:
            print(e)
            #save the invention which caused the error
            with open('d2l_errors.txt', 'w') as f:
                print("Error in parsing abstraction:", invention.body, e)
                f.write(invention.body + "\n") 

            continue

        arity = len(bound_variables_dict)
        operations, single_operation_invention = contains_only_one_operation(lambdabeam_program, base_function_dict)
        if arity == 0:
            if "x1" not in lambdabeam_program:
                try:
                    evaluation = run_program(lambdabeam_program, {"x1": [[1]]}, dc_prims, old_inventions)[0]
                    #if isinstance(evaluation, int):
                    print("EVALUATION OF NON VARIABLE ABSTRACTION: ", evaluation)
                    if evaluation not in domain.constants:
                        print("FOUND A NEW CONSTANT")
                        domain.constants.append(evaluation)
                    print("CONSTANTS: ", domain.constants)
                    print("----------------------")
                    continue
                except:
                    continue

        if single_operation_invention and contains_no_constants(lambdabeam_program) and pruning and abstraction_refinement:
            skipped_inventions[invention.name] = list(operations.keys())[0]

        for old_inv in old_inventions:
            if lambdabeam_program == old_inv.program:
                skipped_inventions[invention.name] = old_inv.name
                break

        if invention.name in skipped_inventions.keys():
            print("SKIPPED ONE: ", lambdabeam_program)
            print("----------------------")
            continue  
            
        base_function_dict[invention.name] = arity
        if sum([item[0] for item in list(bound_variables_dict.values())]) > 0:
            # get only the first item of the tuples in bound_variables_dict.values()
            higher_order_functions[invention.name] = [item[0] for item in list(bound_variables_dict.values())]
            
        print("FOUND ONE: ", lambdabeam_program)
        outer_op = get_outer_op(lambdabeam_program, base_function_dict)
        new_op = Invented(arity, lambdabeam_program, invention, bound_variables = [item[0] for item in list(bound_variables_dict.values())], dc_prims = dc_prims, inventions= old_inventions)
        old_inventions.append(new_op)
        if model is not None and optimizer is not None:
            func_for_args = [item[1] for item in list(bound_variables_dict.values())]
            # flatten func_for_args
            func_for_args = [item for sublist in func_for_args for item in sublist]
            model.add_invention(invention, outer_op, func_for_args = func_for_args, device = device, initialization_method = initialization_method)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        added = True
        dc_counter += 1
        
        # Add to domain:
        domain.operations.append(new_op)

        # add to dc abstractions
        dc_abstractions.append(invention)

        print("----------------------")


        
    return old_inventions, higher_order_functions, base_function_dict, optimizer, added
        

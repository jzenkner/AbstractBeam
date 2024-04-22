import re
import ast
import sys 
sys.path.append("../../")
from crossbeam.dsl.deepcoder_utils import simplify, generate, tokenize, parse
from crossbeam.abstraction.lambdabeam2dreamcoder import *

def convert(program, function_dict, higher_order_functions):
    tokens = [token for token in re.findall(r'-?\b\w+\b|[():\[\]]', program) if token.strip()]
    highest_var_number = 0
    for i, char in enumerate(program):
        if char.isdigit() and program[i-1] == "x":
            if int(char) > highest_var_number:
                highest_var_number = int(char)
    for token in tokens:
        if token.isalpha() and len(token) == 1:
            # replace token in list
            replacement = "x" + str(highest_var_number + 1)
            tokens = [replacement if x == token else x for x in tokens]
            highest_var_number += 1

    dreamcoder_program = "(lambda "
    vars_assignment = False 
    vars_counter = 0
    global_vars_counter = 0
    open_paranthesis = 0
    skip_closing_paranthesis= False
    stack = [(0,0)]
    arguments_counter = [["blank", 0]]
    for i, token in enumerate(tokens):
        if token in function_dict.keys() and token not in [str(i) for i in range(-1, 5)]:
            already_one_var = False
            dreamcoder_program += "(" + token + " " # upper case
            arguments_counter[-1][1] -= 1
            arguments_counter.append([token, function_dict[token]])


        elif token == "lambda":
            already_one_var = False
            if not arguments_counter[-1][0] in higher_order_functions.keys() and arguments_counter[-1][0] == 'lambda':
                vars_assignment = True
            else:
                set_counter = True
                vars_counter = 0
                dreamcoder_program += "(lambda" + " "
                vars_assignment = True
                arguments_counter[-1][1] -= 1
                arguments_counter.append(["lambda", 1])


        elif bool(re.compile(r'^[a-zA-Z]\d+$').match(token)) and vars_assignment:
            if already_one_var:
                dreamcoder_program += "(lambda "
                arguments_counter[-1][1] -= 1
                arguments_counter.append(["lambda", 1])
            already_one_var = True
            if set_counter:
                global_vars_counter+=1
                vars_counter+= 1

        elif token == "[":
            dreamcoder_program += " empty "
            arguments_counter[-1][1] -= 1
            

        elif bool(re.compile(r'^x\d+$').match(token)) and not vars_assignment:
            already_one_var = False
            dreamcoder_program += "$" + str(int(token[1:]) + global_vars_counter - 1)
            if not tokens[i+1] == ")":
                dreamcoder_program += " "
            arguments_counter[-1][1] -= 1
            if len(arguments_counter)>0 and arguments_counter[-1][1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()


        elif bool(re.compile(r'^[a-zA-Z]\d+$').match(token)) and not vars_assignment:
            already_one_var = False
            if tokens[i-2] == ")" and tokens[i-1] == "(" and tokens[i+1] == ")":
                continue
            dreamcoder_program += "$" + str(int(token[1:]) - 1)
            if not tokens[i+1] == ")":
                dreamcoder_program += " "
            arguments_counter[-1][1] -= 1
            
            if len(arguments_counter)>0 and arguments_counter[-1][1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()


        elif token == ")":
            already_one_var = False
            if skip_closing_paranthesis:
                skip_closing_paranthesis = False
                continue
            if len(stack)>0 and stack[-1][0] == open_paranthesis:
                global_vars_counter -= stack[-1][1]
                stack.pop()
            if len(arguments_counter)>0 and arguments_counter[-1][1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()
            open_paranthesis -= 1


        elif token == ":":
            already_one_var = False
            if set_counter:
                stack.append((open_paranthesis, vars_counter))
            vars_assignment = False
            set_counter = False


        elif token == "(":
            already_one_var = False
            open_paranthesis += 1
            
            
        else:
            if token == "]":
                continue
            already_one_var = False
            dreamcoder_program += token + " "
            arguments_counter[-1][1] -= 1
            if len(arguments_counter)>0 and arguments_counter[-1][1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()
            
    dreamcoder_program += ")"
    return dreamcoder_program


def parse_task_string(task_string):
    # Define a regular expression pattern to match the function call
    pattern = r"Task\(\s*name=(.*?),\s*inputs_dict=(.*?),\s*outputs=(.*?),\s*solution=(.*?)\)"

    # Use re.search to find the matches in the input string
    match = re.search(pattern, task_string, re.DOTALL)

    if match:
        # Extract the values of name, inputs, and outputs from the match groups
        name = ast.literal_eval(match.group(1).strip())
        inputs_str = match.group(2).strip()
        outputs_str = match.group(3).strip()

        # Remove line breaks from the inputs and outputs strings
        inputs_str = inputs_str.replace("\n", "")
        outputs_str = outputs_str.replace("\n", "")

        # Convert the inputs and outputs strings to Python objects
        inputs = ast.literal_eval(inputs_str)
        outputs = ast.literal_eval(outputs_str)

        return name, inputs, outputs
    else:
        # Return None if no match is found
        return None


def combine_inputs_outputs(inputs, outputs):
    result = []
    num_inputs = len(inputs)
    
    for i in range(len(outputs)):
        combined_input = tuple(inputs[f'x{j+1}'][i] for j in range(num_inputs))
        result.append((combined_input, outputs[i]))
    
    return result


def build_compression_programs(json_dict, base_function_dict, higher_order_functions, frontiers, top_k=2):
    for solution in json_dict["results"]:
        if solution['solution'] != None:
            task_string = solution["task"]

            task_ast = ast.parse(task_string)

            # Extract the task name from the AST
            task_name = None
            for node in ast.walk(task_ast):
                if isinstance(node, ast.keyword) and node.arg == 'name':
                    task_name = node.value.s

            if task_name not in frontiers.keys():
                frontiers[task_name] = [(solution['solution'], solution["solution_weight"])]
            elif len(frontiers[task_name]) < top_k:
                if solution['solution'] != frontiers[task_name][0][0]:
                    frontiers[task_name].append((solution['solution'], solution["solution_weight"]))
            else:
                max_weight = max([x[1] for x in frontiers[task_name]])
                if solution["solution_weight"] < max_weight:
                    max_index = [x[1] for x in frontiers[task_name]].index(max_weight)
                    frontiers[task_name][max_index] = (solution['solution'], solution["solution_weight"])
    
    # get all solutions in a list
    programs = []
    for task_name in frontiers.keys():
        for solution in frontiers[task_name]:
            programs.append((task_name, solution[0]))
    
    compression_programs = [] 
    tasks = []
    for task_name, program in programs:
        try:
            compression_program = convert(simplify(generate(parse(tokenize(program)))[1:-1]), base_function_dict, higher_order_functions)
        except Exception as e:
            # save current solution in a file for later debugging
            print("Error in converting solution to dreamcoder program:", solution['solution'])
            with open('l2d_errors.txt', 'w') as f:
                f.write(solution['solution'])
            continue

        if compression_program == None:
            print("Error in converting solution to dreamcoder program:", solution['solution'])
            with open('l2d_errors.txt', 'w') as f:
                f.write(solution['solution'])
            continue

        compression_programs.append(compression_program)
        tasks.append(task_name)
    return compression_programs, tasks, frontiers


def find_production_by_name(name, productions):
    for row in productions:
        _, _, function = row
        if str(function) == name:
            return row
    return None

def extract_nested_content(tokens):
    result = []
    stack = []

    for token in tokens:
        if token == '(':
            stack.append(token)
            current_sublist = []
        elif token == ')':
            if stack:
                stack.pop()
                if not stack:
                    result.append(current_sublist)
            else:
                # If ')' is encountered without a matching '(', it's ignored
                pass
        elif stack:
            current_sublist.append(token)

    return result[0] if result else []


def count_arity(signature):
    arity = 0
    no_count = 0
    for char in signature:
        #print(char, no_count)
        if char == "(":
            no_count-= 1
        elif char == ")":
            no_count+= 1
        elif char == ">" and no_count >= 0:
            arity+= 1
    return arity


# Turn in to grammars
from dreamcoder.type import *
config = get_config()
domain = domains.get_domain(config.domain)
operations = domain.operations
primitives = []
for operation in operations:
    primitives.append(Primitive(operation.name, operation.get_signature(), operation))

for j in range(-1, 5):
    primitives.append(Primitive(str(j), tint, j))

baseGrammar = Grammar.uniform([p for p in primitives])


# MAIN
dream_to_lambda_ops = {v[0]: k for k, v in function_dict.items()}

import json

with open('output_original.json', 'rb') as f:
    abstracted_programs = json.load(f)

frontiers = []
for solution in abstracted_programs["results"]:
    if solution['solution'] != None:
        print(solution)
        entry = FrontierEntry(convert(solution['solution'], function_dict), logLikelihood=0, logPrior=0)
        examples = None
        frontiers.append(Frontier([entry], Task(task['name'], examples, None)))












## SAVER

def get_operation_dict():
    operation_dict = {
        "+": Add(),
        "-": Subtract(),
        "*": Multiply(),
        "gt?": Greater(),
        "eq?": Equal(),
        "if": If(),
        "fold": Fold(),
        "map": Map(),
        "empty?": IsEmpty(),
        "cons": Cons(),
        "car": Car(),
        "cdr": Cdr(),
        "is-square": IsSquare(),
        "is-prime": IsPrime(),
        "length": Length(),
        "index": Index(),
        "mod": Mod(),
    }
    return operation_dict
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


def convert(program, function_dict):
    tokens = [token for token in re.findall(r'-?\b\w+\b|[():]', program) if token.strip()]
    dreamcoder_program = "(lambda "
    vars_assignment = False 
    vars_counter = 0
    global_vars_counter = 0
    open_paranthesis = 0
    skip_closing_paranthesis= False
    stack = [(0,0)]
    arguments_counter = [0]
    lambda_func = False
    for i, token in enumerate(tokens):
        if token in function_dict.keys() and token not in [str(i) for i in range(-1, 5)]:
            dreamcoder_program += "(" + function_dict[token][0] + " "
            arguments_counter[-1] -= 1
            arguments_counter.append(function_dict[token][1])
            if token in ["Fold", "Map"]:
                lambda_func = True
            else:
                lambda_func = False


        elif token == "lambda":
            if not lambda_func:
                vars_assignment = True
            else:
                set_counter = True
                vars_counter = 0
                dreamcoder_program += "(" + token + " "
                vars_assignment = True
                arguments_counter[-1] -= 1
                lambda_func = False
                arguments_counter.append(1)


        elif bool(re.compile(r'^[a-zA-Z]\d+$').match(token)) and vars_assignment:
            if set_counter:
                global_vars_counter+=1
                vars_counter+= 1


        elif bool(bool(re.compile(r'^x\d+$').match(token))) and not vars_assignment:
            dreamcoder_program += "$" + str(int(token[1:]) + global_vars_counter - 1)
            if not tokens[i+1] == ")":
                dreamcoder_program += " "
            arguments_counter[-1] -= 1
            lambda_func = False
            while len(arguments_counter)>0 and arguments_counter[-1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()


        elif bool(re.compile(r'^[a-zA-Z]\d+$').match(token)) and not vars_assignment:
            if tokens[i-2] == ")" and tokens[i-1] == "(" and tokens[i+1] == ")":
                continue
            dreamcoder_program += "$" + str(int(token[1:]) - 1)
            if not tokens[i+1] == ")":
                dreamcoder_program += " "
            arguments_counter[-1] -= 1
            lambda_func = False
            
            while len(arguments_counter)>0 and arguments_counter[-1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()


        elif token == ")":
            if skip_closing_paranthesis:
                skip_closing_paranthesis = False
                continue
            if len(stack)>0 and stack[-1][0] == open_paranthesis:
                global_vars_counter -= stack[-1][1]
                stack.pop()
            while len(arguments_counter)>0 and arguments_counter[-1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()
            open_paranthesis -= 1


        elif token == ":":
            if set_counter:
                stack.append((open_paranthesis, vars_counter))
            vars_assignment = False
            set_counter = False


        elif token == "(":
            open_paranthesis += 1
            
            
        else:
            dreamcoder_program += token + " "
            arguments_counter[-1] -= 1
            lambda_func = False
            while len(arguments_counter)>0 and arguments_counter[-1] == 0:
                dreamcoder_program += ")"
                arguments_counter.pop()
            
        #print("----")
    dreamcoder_program += ")"
    #print(dreamcoder_program)
    #print("Program after parse ", Program.parse(dreamcoder_program))
    #print("=====================================")
    return Program.parse(dreamcoder_program)



def induceGrammar_Beta(g0, frontiers, _=None,
                       pseudoCounts=1.,
                       a=3,
                       aic=1.,
                       topK=2,
                       topI=50,
                       structurePenalty=1.,
                       CPUs=1):
    """grammar induction using only version spaces"""
    from dreamcoder.fragmentUtilities import primitiveSize
    import gc
    
    originalFrontiers = frontiers
    frontiers = [frontier for frontier in frontiers if not frontier.empty]
    eprint("Inducing a grammar from", len(frontiers), "frontiers")

    arity = a

    def restrictFrontiers():
        return parallelMap(CPUs,#CPUs,
                           lambda f: g0.rescoreFrontier(f).topK(topK),
                           frontiers,
                           memorySensitive=True,
                           chunksize=1,
                           maxtasksperchild=1)
    restrictedFrontiers = restrictFrontiers()
    def objective(g, fs):
        ll = sum(g.frontierMDL(f) for f in fs )
        sp = structurePenalty * sum(primitiveSize(p) for p in g.primitives)
        return ll - sp - aic*len(g.productions)
            
    v = None
    def scoreCandidate(candidate, currentFrontiers, currentGrammar):
        try:
            newGrammar, newFrontiers = v.addInventionToGrammar(candidate, currentGrammar, currentFrontiers,
                                                               pseudoCounts=pseudoCounts)
        except InferenceFailure:
            # And this can occur if the candidate is not well typed:
            # it is expected that this can occur;
            # in practice, it is more efficient to filter out the ill typed terms,
            # then it is to construct the version spaces so that they only contain well typed terms.
            return NEGATIVEINFINITY
            
        o = objective(newGrammar, newFrontiers)

        #eprint("+", end='')
        eprint(o,'\t',newGrammar.primitives[0],':',newGrammar.primitives[0].tp)

        # eprint(next(v.extract(candidate)))
        # for f in newFrontiers:
        #     for e in f:
        #         eprint(e.program)
        
        return o
        
    with timing("Estimated initial grammar production probabilities"):
        g0 = g0.insideOutside(restrictedFrontiers, pseudoCounts)
    oldScore = objective(g0, restrictedFrontiers)
    eprint("Starting grammar induction score",oldScore)
    
    while True:
        v = VersionTable(typed=False, identity=False)
        with timing("constructed %d-step version spaces"%arity):
            versions = [[v.superVersionSpace(v.incorporate(e.program), arity) for e in f]
                        for f in restrictedFrontiers ]
            eprint("Enumerated %d distinct version spaces"%len(v.expressions))
        
        # Bigger beam because I feel like it
        candidates = v.bestInventions(versions, bs=3*topI)[:topI]
        eprint("Only considering the top %d candidates"%len(candidates))

        # Clean caches that are no longer needed
        v.recursiveTable = [None]*len(v)
        v.inhabitantTable = [None]*len(v)
        v.functionInhabitantTable = [None]*len(v)
        v.substitutionTable = {}
        gc.collect()
        
        with timing("scored the candidate inventions"):
            scoredCandidates = parallelMap(CPUs, # CPUs,
                                           lambda candidate: \
                                           (candidate, scoreCandidate(candidate, restrictedFrontiers, g0)),
                                            candidates,
                                           memorySensitive=True,
                                           chunksize=1,
                                           maxtasksperchild=1)
        if len(scoredCandidates) > 0:
            bestNew, bestScore = max(scoredCandidates, key=lambda sc: sc[1])
            
        if len(scoredCandidates) == 0 or bestScore < oldScore:
            eprint("No improvement possible.")
            # eprint("Runner-up:")
            # eprint(next(v.extract(bestNew)))
            # Return all of the frontiers, which have now been rewritten to use the
            # new fragments
            frontiers = {f.task: f for f in frontiers}
            frontiers = [frontiers.get(f.task, f)
                         for f in originalFrontiers]
            return g0, frontiers
        
        # This is subtle: at this point we have not calculated
        # versions bases for programs outside the restricted
        # frontiers; but here we are rewriting the entire frontier in
        # terms of the new primitive. So we have to recalculate
        # version spaces for everything.
        with timing("constructed versions bases for entire frontiers"):
            for f in frontiers:
                for e in f:
                    v.superVersionSpace(v.incorporate(e.program), arity)
        newGrammar, newFrontiers = v.addInventionToGrammar(bestNew, g0, frontiers,
                                                           pseudoCounts=pseudoCounts)
        eprint("Improved score to", bestScore, "(dS =", bestScore-oldScore, ") w/ invention",newGrammar.primitives[0],":",newGrammar.primitives[0].infer())
        oldScore = bestScore

        for f in newFrontiers:
            eprint(f.summarizeFull())

        g0, frontiers = newGrammar, newFrontiers
        restrictedFrontiers = restrictFrontiers()




    def annotate_types(invention):
    lambda_counter = 0
    parentheses_stack = []
    parentheses_counter = 0
    tokens = [token for token in re.findall(r'-?\b\w+\b|\$\d|[():]', invention) if token.strip()]
    for i, token in enumerate(tokens):
        if token in ["fold", "map"]:
            lambda_counter += 1
            parentheses_stack.append(parentheses_counter)
        elif token[0] == "$" and token[1:].isdigit():

            if int(token[1:]) >= lambda_counter:
                invention = invention.replace(token, "g" + str(int(token[1:]) - lambda_counter), 1)
            else:
                invention = invention.replace(token, "l" + token[1:], 1)
        elif token == "(":
            parentheses_counter += 1
        elif token == ")":
            parentheses_counter -= 1
            if len(parentheses_stack)>0 and parentheses_counter == parentheses_stack[-1]:
                parentheses_stack.pop()
                lambda_counter -= 1
            
    return invention

def parse_to_lambdabeam(program, dream_to_lambda_ops):
    program = annotate_types(program)
    parentheses_stack = []
    parentheses_counter = 0
    lambdaBeam_program = ""
    tokens = [token for token in re.findall(r'\b\w+\?|-?\b\w+\b|\$\d|[():+*-]', program) if token.strip()]
    for i, token in enumerate(tokens):
        if token in dream_to_lambda_ops.keys():
            if token in ["fold", "map"]:
                lambdaBeam_program += dream_to_lambda_ops[token] + "(" + "lambda v1: "
                parentheses_stack.append(parentheses_counter)
            else:
                lambdaBeam_program += dream_to_lambda_ops[token] + "("
                parentheses_stack.append(parentheses_counter)
            
        elif token[0] == "g" and token[1:].isdigit():
            lambdaBeam_program += "x" + str(int(token[1:]) + 1)
            if tokens[i + 1] != ")":
                lambdaBeam_program += ", "

        elif token[0] == "l" and token[1:].isdigit():
            lambdaBeam_program += "v" + str(int(token[1:]) + 1)
            if tokens[i + 1] != ")":
                lambdaBeam_program += ", "

        elif token.isdigit():
            lambdaBeam_program += token
            if tokens[i + 1] != ")":
                lambdaBeam_program += ", "


        elif token == "(":
            parentheses_counter += 1
            
        elif token == ")":
            if len(parentheses_stack) > 0 and parentheses_stack[-1] == parentheses_counter:
                parentheses_stack.pop()
                lambdaBeam_program += ")"
                if i != len(tokens) - 1 and (tokens[i + 1] != ")" or not (len(parentheses_stack) > 0 and parentheses_stack[-1] == parentheses_counter - 1)):
                    lambdaBeam_program += ", "
            parentheses_counter -= 1
    
    # switch arguments since DreamCoder is switching it due to the way it is parsing the program
    if "x1" in lambdaBeam_program and "x2" in lambdaBeam_program:
        lambdaBeam_program = lambdaBeam_program.replace("x1", "temp").replace("x2", "x1").replace("temp", "x2")

    if lambdaBeam_program[-2:] == ", ":
        return lambdaBeam_program[:-2]
    else:
        return lambdaBeam_program

class Invented(DeepCoderOperation):
  def __init__(self, arity, program, signature): 
    super(Invented, self).__init__(arity)
    self.program = program
    self.signature = signature
    self.arity = arity

  def apply_single(self, raw_args):
    arguments = raw_args
    if len(arguments) != self.arity:
      return None
    
    input_dict = {f"x{i+1}" : [arg] for i, arg in enumerate(arguments)}
    return run_program(self.program, input_dict)[0]


  def get_signature(self):
    """The types of this operation's arguments, or None to allow any types."""
    return self.signature
  


  def get_arity(input_string):
    in_brackets = 0
    return sum(1 for char in input_string if (in_brackets := in_brackets + (char == '(') - (char == ')')) == 0 and char == '-')

def has_higher_order_function(input_string):
    in_brackets = 0
    for char in input_string:
        if char == '(':
            in_brackets += 1
        elif char == ')':
            in_brackets = max(0, in_brackets - 1)
        elif char == '-' and in_brackets > 0:
            return True
    return False



# MAIN
import json
import sys
import re
import ast
sys.path.append("../../")
from dreamcoder.frontier import FrontierEntry, Frontier
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.domains.list.listPrimitives import LambdaBeamPrimitives
from crossbeam.dsl import task as task_module
from dreamcoder.program import Program
from dreamcoder.type import TypeConstructor
from dreamcoder.program import Primitive
from dreamcoder.type import *
from dreamcoder.domains.list.listPrimitives import _map
from dreamcoder.grammar import *
from dreamcoder.vs import *
from crossbeam.dsl.deepcoder_operations import *
from crossbeam.dsl.deepcoder_utils import run_program, simplify

dream_to_lambda_ops = {v[0]: k for k, v in function_dict.items()}

import json
# Read the JSON file and load it into a Python dictionary
with open('output_original.json', 'rb') as f:
    abstracted_programs = json.load(f)

frontiers = []
for solution in abstracted_programs["results"]:
    if solution['solution'] != None:
        entry = FrontierEntry(convert(solution['solution'], function_dict), logLikelihood=0, logPrior=0)
        name, input, output = parse_task_string(solution['task'])
        examples = combine_inputs_outputs(input, output)
        request =  arrow(tlist(tint), tlist(tint))
        frontiers.append(Frontier([entry], Task(name=name, request=request, examples=examples)))


if False:
    g, newFrontiers = induceGrammar_Beta(baseGrammar, frontiers[1:],
                        CPUs=15,
                        a=3,
                        structurePenalty=0.)
    invented_productions = [production for production in g.productions if str(production[2])[0] == "#"]

inventions = []
for invention in invented_productions:
    dc_invention = str(invention[2])
    dc_signature = str(invention[1])
    if has_higher_order_function(dc_invention):
        print(invention[2])
        continue
    
    arity = get_arity(str(dc_signature))
    lambdabeam_program = parse_to_lambdabeam(dc_invention, dream_to_lambda_ops)
    inventions.append(Invented(arity, lambdabeam_program, dc_signature))
    # TODO: Now add to operations



"""            
 """
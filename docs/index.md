
# AbstractBeam: Enhancing Bottom-Up Program Synthesis Using Library Learning
<table align="center" border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="./tower_task_102_44.png" alt="Target" style="display: block; margin: auto;">
      <div style="text-align: center;">Target</div>
    </td>
    <td style="text-align: center; border: none;">
      <img src="./tower_construction.gif" alt="Animation of Construction Process" style="display: block; margin: auto;">
      <div style="text-align: center;">Animation of Construction Process</div>
    </td>
    <td style="text-align: center; border: none;">
      <img src="./searchtree.png" alt="Search Tree with depth=4" style="display: block; margin: auto;">
      <div style="text-align: center;">Search Tree with depth=4</div>
    </td>
  </tr>
</table>

## Abstract
LambdaBeam is an execution-guided algorithm for program synthesis that efficiently generates programs using higher-order functions, lambda functions, and iterative loops within a Domain-Specific Language (DSL). However, it does not capitalize on recurring program blocks commonly found in domains like list traversal. 

To address this, *AbstractBeam* introduces *Library Learning*, which identifies and integrates recurring program structures into the DSL, optimizing the synthesis process. Experimental results show that AbstractBeam significantly outperforms LambdaBeam in terms of task completion and efficiency, reducing the number of candidate programs and the time required for synthesis. Library Learning proves beneficial even in domains not explicitly designed for it, demonstrating its broad applicability.

Together, these advancements showcase how AbstractBeam leverages Library Learning to improve upon traditional synthesis methods, balancing the strengths of execution-guided search and reusable code abstractions.

## List Manipulations
![image](https://github.com/user-attachments/assets/1356c2a7-a149-408c-9be5-4da561bdfcef)
The list manipulation domain on which AbstractBeam was evaluated consists of a set of programming tasks focused on processing and transforming lists of integers. These tasks, commonly encountered in functional programming, involve operations such as filtering, mapping, and folding elements of a list based on specific criteria. For instance, tasks may require the generation of programs that double all even numbers in a list, filter out odd numbers, or compute aggregate values like sums or counts based on conditions. The complexity of these tasks grows with the introduction of loops, conditional statements, and higher-order functions. 

## Problem Statement
A major challenge in program synthesis is the **exploding search space**, which grows exponentially as the size of the DSL or the length of the programs increases. When synthesizing programs, each additional operation or program block adds to the depth and breadth of the search tree, causing a combinatorial explosion. For example, in a large DSL with many available operations, the number of possible combinations of these operations becomes unmanageable, particularly as the program length grows. This makes it increasingly difficult to find correct programs within a reasonable amount of time. The trial-and-error approach to testing every combination becomes impractical for complex tasks, leading to a bottleneck in the efficiency and scalability of traditional synthesis methods. Optimizing the search process, such as through techniques like Library Learning in AbstractBeam, is crucial for reducing the search depth by identifying reusable components and narrowing down the number of candidate programs, making synthesis feasible even in more extensive DSLs or longer programs.


This repository contains the source code associated with this [preprint](https://arxiv.org/abs/2405.17514):


## Code
### Setup
Make sure to install pytorch and pytorch-scatter.
We used torch==2.2.1 and torch-scatter==2.1.2 and python=3.8.19.
Then install the packages in the requirements file, e.g.:
```
pip install -r requirements.txt
```
### File structure

The synthetic training data is saved to  `./neurips/abstractbeam/data` and  `./neurips/lambdabeam/data`.
Make sure to also create `./neurips/abstractbeam/models` and `./neurips/abstractbeam/results` directories. Same goes when you want to train the LambdaBeam benchmark.

### Train or eval the model
Navigate to `crossbeam/experiment/deepcoder` directory, and select the config you want to run.
Just adapt the config path to point to `./crossbeam/experiment/deepcoder/configs/` + [`train/abstractbeam.py`, `train/baseline.py`, `eval/abstractbeam_eval.py`, `eval/lambdabeam_eval.py`].
You can make any necessary edits to the selected config file including the data, model, and result directories.
Moreover, you can adapt the hyperparameters, e.g., the enumeration timeout, number of GPUs to use, ... .
To start run below from the project's root (the number of GPUs set in the config file must align with the number selected in below script):

```
./crossbeam/experiment/deepcoder/run_deepcoder.sh
```


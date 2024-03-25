import random
import numpy as np
import pandas as pd
from challenge_basic import get_data, get_data2
import sys

x_train, t_train, x_test, t_test = None, None, None, None
input_filename = None


def gen_input_output():
    """
    Generate input-output pairs from a sequence of data.

    Parameters:
        `data` - a list of integers representing a sequence of notes

    Returns: a list of pairs of the form (x, t) where
        `x` - a numpy array of shape (20, 128) representing the input
        `t` - an integer representing the target note
    """
    global x_train, t_train, x_test, t_test, input_filename
    x_train, t_train, x_test, t_test = get_data2(input_filename)


if __name__ == "__main__":
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    import sys
    from itertools import permutations


    if len(sys.argv) != 6:
        print("Usage: python model_dist.py <start_num_layers> <end_num_layers> <min_hidden_units> <max_hidden_units> <input_file_name> <output_file_name>")
        sys.exit(1)
    
    try:
        start_num_layers = int(sys.argv[1])
        end_num_layers = int(sys.argv[2])
        min_hidden_units = int(sys.argv[3])
        max_hidden_units = int(sys.argv[4])
    except ValueError:
        print("Invalid input")
        sys.exit(1)

    input_filename = sys.argv[5]
    output_filename = sys.argv[6]

    gen_input_output()

    with open(output_filename, "w") as f:
        f.write(f'Training models from {start_num_layers} to {end_num_layers} layers with hidden units from {min_hidden_units} to {max_hidden_units}\n')
    
    for i in range(start_num_layers, end_num_layers+1):
        for perm in permutations(range(min_hidden_units, max_hidden_units), i):
            mlp = MLPClassifier(hidden_layer_sizes=perm, max_iter=3000)
            mlp.fit(x_train, t_train)
            score = mlp.score(x_test, t_test)
            print("Training done with score: ", score)
            if score > max:
                max = score
                max_perm = perm
                print(f"Max score: {max} for {max_perm}")
                with open(output_filename, "a") as f:
                    f.write(f"Max score: {max} for {max_perm}\n")

    
           
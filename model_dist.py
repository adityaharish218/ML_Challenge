import random
import numpy as np
import pandas as pd
from challenge_basic import get_data, get_data2
import sys
from itertools import product

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




def generate_valid_products(min_hidden_units, max_hidden_units, step, leftmost_number_start, leftmost_number_end, layer_size) -> list:
    """
    Generate valid products of hidden units for a given number of layers.

    Parameters:
        `min_hidden_units` - an integer representing the minimum number of hidden units
        `max_hidden_units` - an integer representing the maximum number of hidden units
        `step1` - an integer representing the step size for the number of layers
        `step2` - an integer representing the step size for the number of hidden units

    Returns: a list of tuples representing valid products of hidden units
    """ 
    valid_ranges_3 = ([5, 100], [5, 150], [5, 50])
    valid_ranges_4 = ([5, 100], [5, 200], [5, 150], [5, 50])

    valid_products = []
    if layer_size == 3:
        for perm in product(range(min_hidden_units, max_hidden_units + 1, step), repeat=layer_size):
            if all([valid_ranges_3[i][0] <= perm[i] <= valid_ranges_3[i][1] for i in range(layer_size)]) and perm[0] >= leftmost_number_start and perm[0] <= leftmost_number_end:
                valid_products.append(perm)
    elif layer_size == 4:
        for perm in product(range(min_hidden_units, max_hidden_units + 1, step), repeat=layer_size):
            if all([valid_ranges_4[i][0] <= perm[i] <= valid_ranges_4[i][1] for i in range(layer_size)]) and perm[0] >= leftmost_number_start and perm[0] <= leftmost_number_end:
                valid_products.append(perm)

    return valid_products





if __name__ == "__main__":
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    import sys
    from itertools import product


    if len(sys.argv) != 10:
        print("Usage: python model_dist.py <start_num_layers> <end_num_layers> <min_hidden_units> <max_hidden_units> <step> <left_most_number_start> <left_most_number_stop> <input_file_name> <output_file_name>")
        sys.exit(1)
    
    try:
        start_num_layers = int(sys.argv[1])
        end_num_layers = int(sys.argv[2])
        min_hidden_units = int(sys.argv[3])
        max_hidden_units = int(sys.argv[4])
        step = int(sys.argv[5])
        left_most_number_start = int(sys.argv[6])
        left_most_number_end = int(sys.argv[7])
    except ValueError:
        print("Invalid input")
        sys.exit(1)

    input_filename = sys.argv[8]
    output_filename = sys.argv[9]

    gen_input_output()

    with open(output_filename, "w") as f:
        f.write(f'Training models from {start_num_layers} to {end_num_layers} layers with hidden units from {min_hidden_units} to {max_hidden_units}\n')
    
    max = 0 
    max_perm = None
    for i in range(start_num_layers, end_num_layers+1):
        count = 0
        for perm in generate_valid_products(min_hidden_units, max_hidden_units, step, left_most_number_start, left_most_number_end, i):
            print(perm)
            count +=1
            # mlp = MLPClassifier(hidden_layer_sizes=perm, max_iter=3000)
            # mlp.fit(x_train, t_train)
            # score = mlp.score(x_test, t_test)
            # print(f"Training done with permutation {perm} and score: {score}")
            # if score > max:
            #     max = score
            #     max_perm = perm
            #     print(f"Max score: {max} for {max_perm}")
            #     with open(output_filename, "a") as f:
            #         f.write(f"Max score: {max} for {max_perm}\n")

        print(count)






    
           
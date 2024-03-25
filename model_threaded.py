import threading
from itertools import permutations
from sklearn.neural_network import MLPClassifier
from challenge_basic import get_data2

x_train, t_train, x_test, t_test = None, None, None, None

def gen_input_output():
    """
    Generate input-output pairs from a sequence of data.

    Parameters:
        `data` - a list of integers representing a sequence of notes

    Returns: a list of pairs of the form (x, t) where
        `x` - a numpy array of shape (20, 128) representing the input
        `t` - an integer representing the target note
    """
    global x_train, t_train, x_test, t_test
    file_name = "/Users/adityaharish/Documents/Documents/Subjects/University/UTM/Year_3/Winter_2024_Courses/CSC311/ML_Challenge/clean_dataset.csv"
    x_train, t_train, x_test, t_test = get_data2(file_name)

def process_permutations(thread_num, x_train, t_train, x_test, t_test, start_index, end_index):
    max_score = 0
    max_perm = None
    
    output_file = f"output_file_{thread_num}.txt"
    
    print("Thread", thread_num, "started")
    with open(output_file, "w") as f:
        f.write("Results with max\n")
    
    for i in range(start_index, end_index):
        for perm in permutations(range(20, 300), i):
            print(perm)
            mlp = MLPClassifier(hidden_layer_sizes=perm, max_iter=300)
            mlp.fit(x_train, t_train)
            score = mlp.score(x_test, t_test)

            print(f"Thread {thread_num}: Score: {score} for {perm}")
            
            if score > max_score:
                max_score = score
                max_perm = perm
                print(f"Thread {thread_num}: Max score: {max_score} for {max_perm}")
                with open(output_file, "a") as f:
                    f.write(f"Max score: {max_score} for {max_perm}\n")

def run_threads(num_threads, x_train, t_train, x_test, t_test):
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=process_permutations, args=(i, x_train, t_train, x_test, t_test, i, i + 1))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

# Example usage:
# Assuming x_train, t_train, x_test, t_test are already defined
gen_input_output()
run_threads(10, x_train, t_train, x_test, t_test)

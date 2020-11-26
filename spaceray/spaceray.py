import time
import ray
from ray import tune
import json
from hyperspace import create_hyperspace
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
from tqdm import tqdm
import sys


def get_trials(args):
    # load hyperspace boundaries from json file
    try:
        f = open(args.json, "r")
    except Exception as e:
        print(e)
        print("ERROR: json file with hyperparameter bounds not found. Please use utilities/generate_hyperspace_json.py "
              "to generate boundary file and try again.")
        sys.exit()
    bounds = json.load(f)
    for n in bounds:
        bounds[n] = tuple(bounds[n])
    hyperparameters = list(bounds.values())
    space = create_hyperspace(hyperparameters)
    return space, bounds


def run_experiment(args, func, mode="max", metric="average_res",
                          ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results"):
    """ Generate hyperparameter spaces and run each space sequentially. """
    start_time = time.time()
    ray.init()
    space, bounds = get_trials(args)
    # Run and aggregate the results
    results = []
    i = 0
    error_name = args.out.split(".csv")[0]
    error_name += "_error.txt"
    error_file = open(error_name, "w")
    for section in tqdm(space):
        optimizer = Optimizer(section)
        search_algo = SkOptSearch(optimizer, list(bounds.keys()), metric=metric, mode=mode)
        try:
            analysis = tune.run(func, search_alg=search_algo, num_samples=args.trials,
                                resources_per_trial={'cpu': 25, 'gpu': 1},
                                local_dir=ray_dir)
            results.append(analysis)
        except Exception as e:
            error_file.write("Unable to complete trials in space " + str(i) + "... Exception below.")
            error_file.write(str(e))
            error_file.write("\n\n")
            print("Unable to complete trials in space " + str(i) + "... Continuing with other trials.")
        i += 1

    print("Measured time needed to run trials: ")
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))

    error_file.close()

    # save results to specified csv file
    all_pt_results = results[0].results_df
    for i in range(1, len(results)):
        all_pt_results = all_pt_results.append(results[i].results_df)

    all_pt_results.to_csv(args.out)
    print("Ray Tune results have been saved at " + args.out + " .")
    print("Error file has been saved at " + error_name + " .")

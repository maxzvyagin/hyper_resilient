# SpaceRay
Integration of HyperSpace with Ray Tune hyperparameter search functionality.

Usage requirements: 
- Definition of objective function which takes a config parameters, a dict supplied by Ray Tune. 
- An `argparse Namespace` object which contains: args.trials (number of trials), args.out (intermediate results directory), and args.json (hyperparameter bounds file location in JSON format). See `example/example.json`. 

Number of total hyperspaces generated and searched is dependent on number of parameters `n`, where the total number of spaces is `2^n`.

Also, if you plan on using the Weight and Biases functionality with logging in your objective function, you must specify your own Weight and Biases API Key. Logging should be performed using the `@wandb.mixin` decorator from Ray Tune.
More information on that can be found here: https://docs.ray.io/en/master/tune/tutorials/tune-wandb.html

Note that this processes that generated hyperspaces concurrently, meaning that end results will need to be manually concatenated or processed individually in the specified results directory. 

For more information on the HyperSpace library, see the original repo: https://github.com/yngtodd/hyperspace. __HyperSpace must be installed in order for SpaceRay to work properly.__

_Note that HyperSpace has some dependencies on previous scikit versions which are not compatible. These dependencies are fixed in my fork of the project at https://github.com/maxzvyagin/hyperspace_.

To install SpaceRay, simply clone and run `pip install .` within the top level directory.

To see an example of how to use the tuning function, check out the `example` folder.

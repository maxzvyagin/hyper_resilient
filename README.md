# SpaceRay
Integration of HyperSpace with Ray Tune hyperparameter search functionality. Requires definition of objective function, number of trials, and hyperparameter bounds in JSON format. Also, if you plan on using the Weight and Biases functionality with logging in your objective function, you must specify your own Weight and Biases API Key. Logging should be performed using the `@wandb.mixin` from Ray Tune. 

To install, simply clone and run `pip install .` within the top level directory.

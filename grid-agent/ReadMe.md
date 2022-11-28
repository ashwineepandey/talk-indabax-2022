# Comparing Q-Learning and Sarsa for Mini-Grid Agent Training

This project utilises the Gym-Minigrid environment to train and test agents using the Q-Learning and Sarsa algorithms. The training script is setup to allow for doing a grid-search across the alpha and gamma hyper-parameters. 
For a deeper understanding of the problem statement, approach and results - refer to accompanying report pdf at `report/MLAI_RL_Gridworld_Report.pdf`.
## Directory Structure

```
grid-agent
    conf #Contains the config.yaml file
    data
        grid-search-results # Containts csv files summarising results
        logs # Logs from various run are stored here
        model-file # Pickle files containing trained q tables are saved here
        performance-artefacts # Graphs monitoring training performance are saved here
        predicted-results # Graphs monitoring prediction performance are saved here
    notebooks # All .ipynb files are stored here
    src # All scripts are stored here
```

## How to run

### Training

The training script is set up to use grid search and can be run as follows:

`cd src`

`python train_gridsearch.py`

* Note: The grid is defined in the config file. If you want to run for a specific set of parameters, ensure you only enter them in the list.

#### Example config.yaml parameter settings
Currently it is set to:
``` 
alpha_range: [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
gamma_range: [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]  
```

It can be set to a specific set as follows:
``` 
alpha_range: [0.2]
gamma_range: [0.8]  
```

### Prediction

The prediction script can be run as follows:

`cd src`
If you would like to use the best QL agent:
`python predict.py --algo ql`
If you would like to use the best Sarsa agent:
`python predict.py --algo sarsa`

Note: Results from prediction are written to logs and will appear in the console as well.
## Additional Notes

- All trained agents during grid-search are saved under the `model-file` directory.
- Agents are stored as pickle files - common object type in python.
- The config file has been pre-loaded with the names of the best agents trained using Q-Learning and Sarsa, using the grid search results csv and sorting by average reward.
- The config file has a parameter `render: False` by default. If you would like to see the render of the environment and the agent's movements, set this to `True`. NOTE: This will considerably increase the execution time for script.  
- To view the comparison of all trained agents' results - you can look at the `data/grid-search-results/comparison.xlsx`

## References
1. R. S. Sutton and A. G. Barto, Reinforcement Learning: An
   Introduction, 2nd ed. The MIT Press, 2018. [Online]. Available:
   http://incompleteideas.net/book/the-book-2nd.html
2. M. Chevalier-Boisvert, L. Willems, and S. Pal, “Minimalistic grid-
   world environment for openai gym,” https://github.com/maximecb/gym-
   minigrid, 2018.

import numpy as np


def create_rngs(seed: int) -> dict:

    # Creates random number generators for the different simulation processes.

    random_vars = ["graph", # initializes random BA graph
                   "opinions", # random human opinions from a [0,1] range 
                   "epsilon", # confidence threshold
                   "bot_attachment", # either uniformly random targeting of nodes or preferentially random to more connected nodes 
                   "attention", # non-uniform random selection of 6 neighbors for attention set based on their weight (activity x numconnections)
                   "rewiring", # randomly chooses whether to unfollow the account with probability Q
                   "replacement" # randomly chooses a new node to connect to 
                   ]

    seed_sequence = np.random.SeedSequence(seed)
    child_seed = seed_sequence.spawn(len(random_vars))

    rngs = {random_vars: np.random.default_rng(child_seed)
        for random_vars, child_seed in zip(random_vars, child_seed)}

    return rngs # returns dict mapping keys in random_vars to Generator objects.
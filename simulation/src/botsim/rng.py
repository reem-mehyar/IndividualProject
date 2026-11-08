import numpy as np


STREAMS = ("graph", # initializes random BA graph
            "opinions", # random human opinions from a [0,1] range 
            "epsilon", # confidence threshold
            "bot_attachment", # either uniformly random targeting of nodes or preferentially random to more connected nodes 
            "attention", # non-uniform random selection of 6 neighbors for attention set based on their weight (activity x numconnections)
            "rewiring", # randomly chooses whether to unfollow the account with probability Q
            "replacement" # randomly chooses a new node to connect to 
             )


def create_rngs(master_seed: int, replicate: int) -> dict[str, np.random.Generator]:

    """ Independent and reproducible random streams for one replicate
    The same (master_seed, replicate) always results in the same identical streams between
    no bot and bot variants, sharing graph, opinion, and epsilon draws. """

    return {name: np.random.default_rng(np.random.SeedSequence(entropy=master_seed, spawn_key=(replicate, i)))
                        for i, name in enumerate(STREAMS)}

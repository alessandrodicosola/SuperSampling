# Stress the model in order to find the batch size for each model architecture

from run_exepriment import run, fix_randomness
import random

if __name__ == "__main__":

    n_experiment = 10

    for n in range(n_experiment):
        # set a new seed for randomness
        random.seed()
        batch_size = random.randint(4, 16)
        n_dabs = random.randint(4, 16)
        n_intra_layers = random.randint(4, 8)

        # fix randomness
        fix_randomness(2020)
        run("TUNING", epochs=1, batch_size=batch_size, n_dabs=n_dabs,
            n_intra_layers=n_intra_layers)

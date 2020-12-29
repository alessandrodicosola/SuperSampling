###
### Find if a particular architecture is able to complete an epoch with a particular batch_size
###


from run_exepriment import run, fix_randomness
import random


def tune(n_dab, n_intra_layers):
    for batch_size in reversed(range(4, 16 + 1, 4)):
        error = run("TUNING", epochs=1, n_workers=0, pin_memory=False, batch_size=batch_size, n_dab=n_dab,
                    n_intra_layers=n_intra_layers)

        # if not error best batch_size is found, no need to reduce it.
        if not error:
            return


if __name__ == "__main__":

    n_experiment = 10

    for _ in range(n_experiment):
        random.seed()
        n_dab = random.randint(4, 16)
        random.seed()
        n_intra_layers = random.randint(4, 8)

        fix_randomness(2020)
        tune(n_dab, n_intra_layers)

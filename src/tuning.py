###
### Find if a particular architecture is able to complete an epoch with a particular batch_size
###


from run_exepriment import run, fix_randomness
import random


def tune(**model_kwargs):
    # fix min batch size to 8 in orer to have at most 100 num batches per epoch such that the training time is not too huge
    for batch_size in reversed(range(8, 16 + 1, 4)):
        error = run("TUNING", epochs=1, n_workers=2, pin_memory=True, batch_size=batch_size, save_checkpoints=1,
                    **model_kwargs)

        # if not error best batch_size is found, no need to reduce it.
        if not error:
            return


def tune_saturate_memory():
    n_experiment = 50

    for _ in range(n_experiment):
        random.seed()
        n_dab = random.randint(4, 8)
        random.seed()
        n_intra_layers = random.randint(4, 8)
        random.seed()
        out_channels_dab = random.randint(16, 64)
        random.seed()
        intra_layer_output_features = random.randint(16, 64)

        fix_randomness(2020)

        tune(n_dab=n_dab, n_intra_layers=n_intra_layers, out_channels_dab=out_channels_dab,
             intra_layer_output_features=intra_layer_output_features)


def tune_lr(**model_kwargs):
    for lr in [1e-2, 1e-3, 1e-4]:
        kwargs = {"lr": lr, **model_kwargs}
        run("LR", epochs=10, n_workers=2, pin_memory=True, batch_size=8, save_checkpoints=1, save=False, **kwargs)


if __name__ == "__main__":
    fix_randomness(2020)
    model_kwargs = {
        "n_dab": 8,
        "n_intra_layers": 4,
        "out_channels_dab": 32,
        "intra_layer_output_features": 32
    }
    tune_lr(**model_kwargs)

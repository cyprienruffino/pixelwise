import os
import datetime

from config import Config
from training import train
from data_io2D import get_texture_iter


def run():
    run_name = str(datetime.datetime.now())
    os.mkdir("../runs/" + run_name)
    print(run_name)
    config = Config(run_name)
    train(
        sgancfg=config,
        data_provider=get_texture_iter("../ti_2D/", batch_size=config.batch_size, npx=config.npx, n_channel=config.nc, mirror=False),
        run_name=run_name,
        checkpoints_dir="../runs/" + run_name + "/checkpoints/",
        logs_dir="../runs/" + run_name + "/logs/",
        samples_dir="../runs/" + run_name + "/samples/",
        use_tensorboard=True,
        plot_models=False)


if __name__ == "__main__":
    run()

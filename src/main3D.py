import os

from training import train
from data_io3D import get_texture_iter
from config import Config
from datetime import datetime

if __name__ == "__main__":
    run_name = str(datetime.now())
    config = Config(run_name)
    os.mkdir("runs/" + run_name)
    print(run_name)
    train(
        sgancfg=config,
        data_provider=get_texture_iter("ti_3D/", batch_size=config.batch_size, npx=config.npx, n_channel=config.nc, mirror=False),
        run_name=run_name,
        checkpoints_dir="runs/" + run_name + "/checkpoints/",
        logs_dir="runs/" + run_name + "/logs/",
        samples_dir="runs/" + run_name + "/samples/",
        use_tensorboard=True,
        plot_models=False)

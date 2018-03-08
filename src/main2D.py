import datetime
import importlib.util
import os
import sys

from datasets.data_io2D import get_texture_iter
from training import train


def run():
    if len(sys.argv) < 2:
        print("Usage : python main2D.py path_to_config_file")
        exit(1)

    run_name = str(datetime.datetime.now())
    print(run_name)
    os.mkdir("./runs/" + run_name)

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config_mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_mod)
    config = config_mod.Config2D(run_name)

    train(
        sgancfg=config,
        data_provider=get_texture_iter("ti_2D/", batch_size=config.batch_size, npx=config.npx),
        run_name=run_name,
        checkpoints_dir="./runs/" + run_name + "/checkpoints/",
        logs_dir="./runs/" + run_name + "/logs/",
        samples_dir="./runs/" + run_name + "/samples/",
        use_tensorboard=True,
        plot_models=False)


if __name__ == "__main__":
    run()

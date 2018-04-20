import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import datetime
import importlib.util
import os
import sys
import shutil
from tftraining import train


def run():
    if len(sys.argv) < 3:
        print("Usage : python main.py path_to_config_file path_to_image_folder [run_name]")
        exit(1)

    if len(sys.argv) > 3:
        run_name = sys.argv[3]
    else:
        run_name = str(datetime.datetime.now())
    print(run_name)
    os.mkdir("./runs/" + run_name)

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(config_module)
    config = config_module.CustomConfig(run_name)

    shutil.copy2(sys.argv[1], './runs/' + run_name + "/config.py")

    train(
        sgancfg=config,
        disc_data_provider=config.disc_data_provider(sys.argv[2], **config.disc_data_provider_args),
        gen_data_provider=config.gen_data_provider(sys.argv[2], **config.gen_data_provider_args),
        run_name=run_name,
        generate_png=False,
        checkpoints_dir="./runs/" + run_name + "/checkpoints/",
        logs_dir="./runs/" + run_name + "/logs/",
        samples_dir="./runs/" + run_name + "/samples/",
        use_tensorboard=True)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run()

import datetime
import importlib.util
import os
import sys
import shutil
from training import train


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
        data_provider=config.data_generator(sys.argv[2], **config.data_gen_args),
        run_name=run_name,
        checkpoints_dir="./runs/" + run_name + "/checkpoints/",
        logs_dir="./runs/" + run_name + "/logs/",
        samples_dir="./runs/" + run_name + "/samples/",
        use_tensorboard=True)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run()

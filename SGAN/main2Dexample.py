from training import train
from data_io2D import get_texture_iter
from config import Config
from datetime import datetime

if __name__ == "__main__":
    run_name = str(datetime.now())
    config = Config(run_name)
    print(run_name)
    train(
        sgancfg=config,
        data_provider=get_texture_iter("ti_2D/", batch_size=config.batch_size),
        run_name=run_name,
        checkpoints_dir="runs/" + run_name + "/checkpoints/",
        logs_dir="runs/" + run_name + "/logs/",
        samples_dir="runs/" + run_name + "/samples/",
        use_tensorboard=True,
        plot_models=False)

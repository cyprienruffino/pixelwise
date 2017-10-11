from training import train
from data_io2D import get_texture_iter
from config import Config

if __name__ == "__main__":
    train(
        sgancfg="demo.sgancfg",
        data_provider=get_texture_iter("ti_2D/"),
        run_name="Demo",
        checkpoints_dir="demo/checkpoints/",
        logs_dir="demo/logs/",
        samples_dir="demo/samples/",
        use_tensorboard=True,
        plot_models=True)

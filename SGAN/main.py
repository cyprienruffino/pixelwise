from training import train

if __name__ == "__main__":
    train(
        sgancfg="demo.sgancgf",
        training_image="ti_2D/ti_2500_1ch",
        run_name="Demo",
        checkpoints_dir="demo/checkpoints",
        logs_dir="demo/logs",
        samples_dir="demo/samples",
        use_tensorboard=True,
        plot_models=True)

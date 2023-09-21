from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 256
    image_channels = 3
    batch_size = 6
    train_batch_size = 6
    eval_batch_size = 6
    num_epochs = 500
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 5
    save_model_epochs = 5
    dataset = 'alkzar90/croupier-mtg-dataset2'
    output_dir = f'models/{dataset.split("/")[-1]}'
    device = "cuda"
    seed = 0
    num_workers = 2
    resume = None


training_config = TrainingConfig()


# params = {
#     "device": "cuda",
#     "lr": 0.001,
#     "batch_size": 32,
#     "num_workers": 2,
#     "epochs": 17, #15
# }
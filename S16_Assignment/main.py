
from config_file import get_config
from train import train_model

config = get_config()
config["batch_size"] = 16
config["preload"] = None
config["num_epochs"] = 10


train_model(config)
import yaml
from trainer import MainTrainer

from constants import *

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_config(os.path.join(CONFIG_DIR, 'Bllossom-8B_train.yaml'))
    trainer = MainTrainer(config)
    trainer.train()



import argparse
import importlib
from trainer import Trainer


def main():
    """ Runs the trainer based on the given experiment configuration """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs.baseline", help='experiment configuration dict')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    trainer = Trainer(config_module.config)
    trainer.run()

if __name__ == "__main__":
    main()
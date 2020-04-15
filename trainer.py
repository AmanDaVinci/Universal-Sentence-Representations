import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.uni_lstm import UniLSTM
from models.bi_lstm import BiLSTM
from models.embedding_encoder import EmbeddingEncoder
from models.classifier import Classifier
from utils import get_dataloaders, accuracy

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
LOG_DIR = Path("logs/")
DATA_DIR = Path("data/")
BEST_MODEL_FNAME = "best-model.pt"


class Trainer():

    def __init__(self, config: dict):
        """
        Initialize the trainer

        Parameters
        ---
        config: dict
        configuration dictionary with the following keys:
        {
            "exp_name"
            "debug"
            "seed"
            "batch_size"
            "epochs"
        }
        """  
        self.config = config
        
        self.exp_dir = RESULTS / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = CHECKPOINTS / config['exp_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        log_name = config["exp_name"]+".log"
        self.logger = logging.getLogger(__name__)
        logfile_handler = logging.FileHandler(filename=self.exp_dir / log_name)
        logfile_handler.setLevel(level = (logging.DEBUG if config["debug"] else logging.INFO))
        logfile_format = logging.Formatter('%(asctime)s - %(levelname)10s - %(funcName)15s : %(message)s')
        logfile_handler.setFormatter(logfile_format)
        self.logger.addHandler(logfile_handler)
        self.logger.setLevel(level = (logging.DEBUG if config["debug"] else logging.INFO))

        print(f"Launched successfully... \nLogs available @ {self.exp_dir / log_name}")
        print("To stop training, press CTRL+C")
        self.logger.info("-"*50)
        self.logger.info(f"EXPERIMENT: {config['exp_name']}")
        self.logger.info("-"*50)

        self.logger.info(f"Setting seed: {config['seed']}")
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.logger.info(f"Loading data ...")
        self.train_dl, self.valid_dl, _, vocab_emb = get_dataloaders(config['batch_size'], DATA_DIR)

         # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0.

        if config['encoder'] == 'EmbeddingEncoder':
            encoded_dim = vocab_emb.shape[-1]
            encoder = EmbeddingEncoder(embeddings=vocab_emb)
        elif config['encoder'] == 'UniLSTM':
            encoded_dim = config['hidden_dim']
            encoder = UniLSTM(embeddings=vocab_emb, 
                              batch_size=config['batch_size'],
                              hidden_size=config['hidden_dim'],
                              num_layers=config['num_layers'])
        elif config['encoder'] == 'BiLSTM':
            encoded_dim = 2*config['hidden_dim']
            encoder = BiLSTM(embeddings=vocab_emb, 
                             batch_size=config['batch_size'],
                             hidden_size=config['hidden_dim'],
                             num_layers=config['num_layers'])
        else:
            self.logger.error("Encoder not available")
            sys.exit(1)

        self.model = Classifier(encoder, encoded_dim)
        self.logger.info(f"Using device: {config['device']}")
        self.model.to(config['device'])
        self.opt = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

        if 'load_checkpoint' in config:
            self.load_checkpoint(config['load_checkpoint'])

    def run(self):
        try:
            self.logger.info(f"Begin training for {self.config['epochs']} epochs")
            self.train()
        except KeyboardInterrupt:
            self.logger.warning("Manual interruption registered. Please wait to finalize...")
            self.save_checkpoint()

    def train(self):
        """ Main training loop """
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            for i, batch in enumerate(self.train_dl):
                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)

                self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                self.logger.debug(f"EPOCH:{epoch} STEP:{i}\t Accuracy: {results['accuracy']:.3f} Loss: {results['loss']:.3f}")

                if i % self.config['valid_freq'] == 0:
                    self.validate()
                if i % self.config['save_freq'] == 0:
                    self.save_checkpoint()

    def validate(self):
        """ Main validation loop """
        self.model.eval()
        losses = []
        accuracies = []

        self.logger.debug("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
            
        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        self.writer.add_scalar('Accuracy/Valid', results['accuracy'], self.current_iter)
        self.writer.add_scalar('Loss/Valid', results['loss'], self.current_iter)
        report = (f"[Validation]\t"
                f"Accuracy: {mean_accuracy:.3f} "
                f"Total Loss: {np.mean(losses):.3f}")
        self.logger.info(report)
                   
    def _batch_iteration(self, batch: tuple, training: bool):
        """ Iterate over one batch """

        # send tensors to model device
        premise = batch.premise[0].to(self.config['device'])
        hypothesis = batch.hypothesis[0].to(self.config['device'])
        premise_seqlen = batch.premise[1]
        hypothesis_seqlen = batch.hypothesis[1]
        label = batch.label.to(self.config['device'])

        if training:
            self.opt.zero_grad()
            pred = self.model((premise, premise_seqlen), (hypothesis, hypothesis_seqlen))
            loss = self.criterion(pred, label)
            loss.backward()
            self.opt.step()
        else:
            with torch.no_grad():
                pred = self.model((premise, premise_seqlen), (hypothesis, hypothesis_seqlen))
                loss = self.criterion(pred, label)

        acc = accuracy(pred, label)
        results = {'accuracy': acc, 'loss': loss.item()}
        return results

    def save_checkpoint(self, file_name: str = None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint directory and checkpoint file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = self.checkpoint_dir / file_name
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, file_name)
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, file_name: str):
        """Load the checkpoint with the given file name

        Checkpoint must contain:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
            - optimizer state

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file
        """
        try:
            file_name = self.checkpoint_dir / file_name
            self.logger.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config['device'])

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")
        
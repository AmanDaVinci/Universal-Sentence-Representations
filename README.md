# Universal Sentence Representations
This project experiments with different ways to learn universal sentence representations. Four different neural models are implemented to encode sentences. These models are trained on the Stanford Natural Language Inference corpus to classify sentence pairs based on their relation. The learned sentence representations are evaluated using the SentEval framework. 
![](architectures.png)

## Project Organization
The project is organized as follows:
* models/ ..................................................... Model Architectures
  * embedding_encoder.py
  * uni_lstm.py
  * bi_lstm.py
  * bi_lstm_pool.py
  * classifier.py
* configs/ ..................................................... Experiment Configurations
  * baseline.py
  * uni_lstm.py
  * bi_lstm.py
  * bi_lstm_pool.py
* checkpoints/ ............................................ Trained Models
* results/ ..................................................... Experiment Logs
* trainer.py ................................................. Training Logic
* utils.py ..................................................... Utilities
* main.py .................................................... Run Experiments
* demo.ipynb .............................................. Demo \& Analysis    

## How to run?
1. Activate conda environment 
```
 conda env create -f "environment.yml
 conda activate prod
```
2. Download the pre-trained checkpoints from [Google Drive](https://drive.google.com/file/d/1fqFlnUrQuvr4U6egrTTcnTbw4e47Nch0/view?usp=sharing)
3. Install [SentEval](https://github.com/facebookresearch/SentEval) 
4. Run the demonstration and analysis notebook: [demo.ipynb](.demo.ipynb)
5. To run our experiments:
```
python3 main.py --config=configs.baseline --train=True --test=True
```
6. To design your own experiment, use the configuration dictionaries in the configs directory as follows:
```
{
  "exp_name": "exp001",                         (str, the name of the experiment, used to save the checkpoints and csv results)
  "device": "cuda:0"/"cpu",                     (str, which device to be used for the model)
  "lr": 1e-4,                                   (float, the learning rate)
  "epochs": 10,                                (int, the number of epochs)
  "batch_size" : 128,                           (int, the size of each batch of data)
  "print_freq": 1000,                            (int, how often to print metrics for the trainint set)
  "eval_freq" : 500,                             (int, how often to evaluate the model and print metrics for the validation set)
  "seed": 42                             (int, the seed to be used for reproducibility)
}
```
## References
1. Conneau, Alexis, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. “Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.” ArXiv:1705.02364 [Cs], July 8, 2018. http://arxiv.org/abs/1705.02364.
2.  Conneau, Alexis, and Douwe Kiela. “SentEval: An Evaluation Toolkit for Universal Sentence Representations.” ArXiv:1803.05449 [Cs], March 14, 2018. http://arxiv.org/abs/1803.05449.

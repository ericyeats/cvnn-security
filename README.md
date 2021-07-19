# cvnn-security
Python/PyTorch code for the ICML 2021 Paper: Improving Gradient Regularization using Complex-Valued Neural Networks

## Requirements
numpy==1.18.4
torch==1.8.0
torchvision==0.9.0

## Workflow
First, train a network using the `train` positional argument, followed by the model architecture positional argument. Optionally specify `--use_complex` and, for gradient regularized training, `--beta BETA`. This will save the trained network, either in the default location or the specified `--net_save_path NET_SAVE_PATH`. Then, one can attack the network using the `attack` positional argument, followed by the model architecture positional argument. Specify the network load path with `--net_load_path NET_LOAD_PATH`. The attack can be configured with optional arguments. One can optionally save the adversarial examples with `--save_examples --examples_save_path EXAMPLES_SAVE_PATH`. Then, evaluate the network (or other network) on the saved examples with the `eval` positional argument, followed by the architecture positional argument. Speciify the network load path with `--net_load_path NET_LOAD_PATH`. Specify if previously stored examples will be used for evaluation: `--eval_adv_examples --examples_load_path EXAMPLES_LOAD_PATH`.

### Train Example
Complex Network
```
python main.py train MNISTNetV1 --use_complex --beta 512 --net_save_path './models/complex_b512.pt'
```

Real Network
```
python main.py train MNISTNetV0 --beta 512 --net_save_path './models/real_b512.pt'  
```

### Attack Example

Complex Network
```
python main.py attack MNISTNetV1 --use_complex --net_load_path './models/complex_b512.pt' --attack_eps 0.1
```
Real Network
```
python main.py attack MNISTNetV0 --net_load_path './models/real_b512.pt' --attack_eps 0.1
```

### Eval Example
```
python main.py eval MNISTNetV1 --use_complex --net_load_path './models/complex_b512.pt' 
```

Cite this work with:
```
@inproceedings{yeats2021improving,
  title={Improving Gradient Regularization using Complex-Valued Neural Networks},
  author={Yeats, Eric C and Chen, Yiran and Li, Hai},
  booktitle={International Conference on Machine Learning},
  pages={11953--11963},
  year={2021},
  organization={PMLR}
}
```
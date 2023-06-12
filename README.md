# AI Systems on V-LoL-Train

This repository serves as a comprehensive analysis of AI systems' visual logical learning abilities on the V-LoL-Train
problem. It includes implementations of various models, including symbolic, neural, and neuro-symbolic approaches,
for tackling visual logical reasoning tasks. The available implementations encompass training, cross-validation, and
plotting functionalities for the following methods:

- Symbolic AI: Popper, Aleph
- Neural AI: ResNet, EfficientNet, Vision Transformer
- Neuro-Symbolic AI: RCNN-Popper, RCNN-Aleph

By utilizing this repository, researchers can assess and compare the performance of different AI systems on the
V-LoL-Train problem, enabling in-depth evaluations of their visual logical reasoning capabilities. The repository
allows to conduct the following experiments:

- evaluation on different V-LoL rules
- evaluation on different V-LoL visualizations
- evaluation on different V-LoL sceneries
- evaluation on different V-LoL train lengths
- evaluation on different degrees of image noise
- evaluation on different degrees of perturbed labels
- evaluation on varying number of training samples
- test-time interventions

# Installation and Setting up docker

We recommend using this repository as a wrapper for the V-Lol-Train dataset generation repository. To do so, please
follow the installation below.
Alternatively, you can also download the dataset from the [official website](https://sites.google.com/view/v-lol).
In that case you need to adjust the dataset path `ds_path` in the script parameters and add the dataloader files 
to the following directory `visual-logical-learning/TrainGenerator/michalski_trains`.
The dataset files can also be found on the [official website](https://sites.google.com/view/v-lol).

```bash
git clone https://github.com/lukashelff/visual-logical-learning.git
cd visual-logical-learning
git clone https://github.com/ml-research/vlol-dataset-gen.git
cd ..
docker build -t v-lol-train .
```

# Usage

To run the experiments, please execute the following command:

```bash
docker run --gpus device=0 -v $(pwd):/home/workdir v-lol-train python3 main.py
```

## Script parameters

The following settings are available, the corresponding input types and default settings are noted in parentheses:

General settings:

- `action` (str, train): The action to perform. Possible values are 'train', 'plot'.
- `command` (str, train): The command to perform. Possible values are 'train', 'eval', 'ilp', 'ilp_crossval',
  'split_ds', 'eval_generalization' or 'ct'.
- `cuda` (int, 0): The cuda device to use. If set to -1, the CPU is used.'

For more information on the available settings, please refer to the code.

# Implementation details

For Popper we set the hyperparameters to allow for a maximum of 10 rules each allowing a maximum of 6 variables and 6
literals in its body. Predicate finding and recursion are turned off, as we could not observe any performance
improvement. For ALEPH we use the following hyperparameters: $clauselength=10$, $minacc=0.6$, $minscore=3$, $minpos=3$,
$nodes=5000$, $explore=true$, $max\_features=10$. Both ILP systems are trained and evaluated on the a symbolic
ground-truths instead of the visual images.

Our subsymbolic models, namely the ResNet, EfficientNet, and Vision Transformer are initialized with the weights of the
pre-trained foundation models which was trained on the 1000-class ImageNet dataset. The last fully-connected layer is
replaced to fit the two-class classification task of westbound and eastbound trains. Subsequently, the models are
transfer trained on the respective datasets for 25 epochs using a batch size of 50 and starting with a learning rate of
0.001 (0.0001 for the Vision Transformer), which decreases by 20\% every five epochs. The Adam optimizer is used for
updating the models' weights and the cross-entropy loss function for calculating the loss.

For the perceptions modules of the Neuro-Symbolic AI systems, we modify the improved mask-RCNN (v2 version)
\cite{li2021benchmarking} to allow for multi-label instance segmentation. For more in depth implementation details
please refer to our code. We initialize our model with pre-trained weights for MaskRCNN + ResNet50 + FPN using the v2
variant with post-paper optimizations. We transfer train our model on 10k V-LoL-Train dataset containing random
trains. For training we use the AdamW optimizer and cross-entropy loss. After inferring the segmented masked using
mask-RCNN we post process these using a mask matching algorithm to assemble a symbolic scene representations. We achieve
nearly 100\% validation accuracy on the random V-LoL-Train and 99\% test accuracy on the Michalski
V-LoL-Train.
Subsequently, we fit the ILP approaches using the same hyperparameters as in the run of the purely symbolic AI systems.
For beam search of $\alpha$ILP we choose a beam size of 70 with a beam depth of 5. We select a maximum of 1000 clauses
after search on which we then perform learning for 100 epochs. For TheoryX and the numerical rule we learn a logic
program consisting of two rules while for the complex rule we learn 4 rules. For more in depth information on the mode
declaration and hyper parameters of $\alpha$ILP, Popper, and Aleph please refer to our code.
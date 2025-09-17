# SynapticNet-A-Continual-Learning-Architecture-with-Dynamic-Growth
A PyTorch implementation of a novel Continual Learning (CL) architecture designed to combat catastrophic forgetting through task-specific parameter masking and a dynamic growth mechanism dubbed "NeuronGenesis".
SynapticNet: A Continual Learning Architecture with Dynamic Growth



A PyTorch implementation of a novel Continual Learning (CL) architecture designed to combat catastrophic forgetting through task-specific parameter masking and a dynamic growth mechanism dubbed "NeuronGenesis".

🧠 Overview
Catastrophic forgetting is a fundamental challenge in neural networks, where learning new tasks causes a rapid degradation of performance on previously learned ones. SynapticNet addresses this by:

Task-Specific Parameter Isolation: Using gating mechanisms to ensure only task-relevant neurons are active during forward and backward passes.

Dynamic Architecture Growth (NeuronGenesis): Automatically expanding the network's capacity by adding new neurons and filters when performance on a new task drops below a threshold, preventing the overwriting of old knowledge.

Multi-Task Inference: A single, unified model capable of performing well on all tasks it has been trained on sequentially.

This project demonstrates the architecture on a sequence of standard vision datasets: MNIST → Fashion-MNIST → CIFAR-10 → CIFAR-100.

✨ Key Features
TaskConv2d & TaskLinear Layers: Custom PyTorch modules that implement task-specific masking for convolutional and linear layers.

Gradient Masking: Applies gradients only to parameters associated with the current task, protecting previously learned weights.

Dynamic Growth API (model.grow()): A method to seamlessly expand the model's width for new tasks, initializing new parameters without disrupting old ones.

Flexible Dataset Handling: The TaskDataset wrapper automatically handles differences in image dimensions (e.g., MNIST to CIFAR) and channels (grayscale to RGB).

📊 Results
The model was trained sequentially on four datasets without revisiting previous data. Final evaluation accuracies are:

Task	Dataset	Accuracy
Task 1	MNIST	98.89%
Task 2	Fashion-MNIST	95.62%
Task 3	CIFAR-10	93.62%
Task 4	CIFAR-100	93.89%
Final Model Size: 16.2 Million parameters

These results demonstrate strong retention on earlier tasks (minimal forgetting on MNIST and Fashion-MNIST) while progressively learning more complex ones.

🚀 Getting Started
Prerequisites
Python 3.8+

PyTorch 2.0+

Torchvision

CUDA-capable GPU (recommended)

Installation
Clone the repository:

bash
git clone https://github.com/your-username/SynapticNet.git
cd SynapticNet
Install required packages:

bash
pip install torch torchvision torchinfo
Usage
The main notebook SynapticNet-CNN-NeuronGenesis.ipynb contains the full code, from model definition to training and evaluation.

Run the Jupyter Notebook:

bash
jupyter notebook SynapticNet-CNN-NeuronGenesis.ipynb
The notebook will automatically download the datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100).

Training Pipeline:
The code follows this sequence:

Initialize the model for Task 1 (MNIST).

Train on Task 1.

Grow the network for Task 2 (Fashion-MNIST).

Train on Task 2.

Repeat the grow-train cycle for Tasks 3 (CIFAR-10) and 4 (CIFAR-100).

Evaluate the final model on all tasks.

🏗️ Architecture
The core components of SynapticNet are:

MultiTaskConvSynapticNet: The main neural network class.

Initialization: Starts with a configurable number of convolutional and dense layers.

.grow() method: Dynamically adds new convolutional channels and hidden neurons for a new task.

.apply_task_gradient_mask(): Freezes gradients for all parameters not related to the current task.

Task-Specific Layers (TaskConv2d, TaskLinear):

Contain a buffer (task_ids) that tags each neuron/filter with its task of origin.

During forward pass, they mask the output of neurons not belonging to the current task.

During backward pass (via apply_gradient_mask), they prevent updates to frozen parameters.

📂 Project Structure

SynapticNet/
├── SynapticNet-CNN-NeuronGenesis.ipynb  # Main Jupyter notebook with full code
├── README.md                            # This file
├── requirements.txt                     # Python dependencies (optional)
└── data/                                # Auto-created directory for datasets
🔮 Future Work / Possible Improvements
Implement a growth trigger: Instead of manual growth, automatically trigger model.grow() when validation accuracy on a new task is below a threshold.

Experiment with growth factors: Tune the number of channels/neurons to add (grow_conv_channels, grow_hidden) based on task complexity.

Add knowledge distillation: Incorporate a distillation loss to further solidify previous task knowledge when learning new tasks.

Benchmark against other CL methods: Compare results with popular methods like EWC, GEM, and Experience Replay.

Explore more complex datasets: Apply the architecture to domains like semantic segmentation or natural language processing.

👨‍💻 Author
Harsha Vardhan Reddy M

LinkedIn: https://www.linkedin.com/in/harsha-vardhan-reddy-maram-reddy-ab9b6b245/

Email: harshavardhanreddy2211@gmail.com

GitHub: https://github.com/Harsha-211

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Inspired by research on Continual Learning and lifelong learning algorithms.

Built with PyTorch.

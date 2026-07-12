# Dl_project_neural_network_using_pytorch.ipynb
# Breast Cancer Classification using PyTorch

This repository contains a complete Deep Learning workflow for classifying breast cancer tumors as malignant or benign using a custom neural network built with PyTorch. 

The project utilizes the Wisconsin Breast Cancer Dataset to train a binary classifier, achieving high accuracy through data standardization and optimized neural network training on a GPU.

## 🚀 Features
* **Data Preprocessing:** Standardizes features using `StandardScaler` to ensure optimal gradient descent convergence.
* **Hardware Acceleration:** Automatically detects and utilizes CUDA (GPU) if available for faster training.
* **Custom Architecture:** Implements a 2-layer fully connected neural network (`nn.Linear`) with ReLU activation for the hidden layer and a Sigmoid activation for binary probability output.
* **Optimized Training:** Trains using Binary Cross-Entropy Loss (`nn.BCELoss`) and the Adam optimizer over 100 epochs.
* **Evaluation Metrics:** Evaluates accuracy on both training and unseen test splits.

## 📊 Performance
* **Training Accuracy:** ~97.36%
* **Final Test Accuracy:** ~97.37%

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3
* **Framework:** PyTorch
* **Machine Learning Tools:** Scikit-Learn (for dataset, data splitting, and preprocessing), NumPy
* 🏗️ Neural Network ArchitectureThe network (NeuralNet) maps 30 input features to a single probability score:Input Layer: 30 features (matching the breast cancer dataset dimensions)Hidden Layer: 64 nodes with ReLU activationOutput Layer: 1 node with Sigmoid activation (outputs a value between 0 and 1)💻 Code Workflow Overview1. Data Collection & PreprocessingPython#
* # Load dataset and split into 80% Train / 20% Test
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
2. Model Initialization & TrainingPython# Hyperparameters
input_size = 30
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop updates weights using backpropagation over 100 epochs
3. EvaluationThe final model performance is verified by rounding the output probabilities to the nearest integer ($0$ or $1$) and comparing them against the true labels.Training Accuracy: 97.36%
Final Test Accuracy: 97.37%

---

### 💡 Tip for your Repo
If you are saving this in your GitHub repository alongside your notebook, you can

To install the required libraries, run:
```bash
pip install torch sklearn numpy

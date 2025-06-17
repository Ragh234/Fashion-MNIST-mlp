# 👚 Fashion MNIST Classification with TensorFlow

A beginner-friendly deep learning project that classifies clothing items from the Fashion MNIST dataset using a Multi-Layer Perceptron (MLP) model in TensorFlow.

# 🧠 Model Overview
- A feedforward neural network built using Keras Sequential API
- Architecture: Flatten → Dense (128, ReLU) → Dense (64, ReLU) → Dense (10, Softmax)
- Optimizer: Adam  
- Loss Function: Sparse Categorical Crossentropy

# 📊 Features Implemented
- Image normalization (scaling pixel values between 0 and 1)
- Validation split from training data
- Model training with accuracy/loss tracking
- Plotting training and validation curves
- Evaluation with confusion matrix & classification report

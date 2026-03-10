#!/usr/bin/env python3
"""
Deep Learning Platform - IBM Professional Certificate Capstone

Provides utilities for building, training, and evaluating deep learning models
using TensorFlow/Keras for image classification tasks.
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    tf = None
    keras = None


class DeepLearningPlatform:
    """Platform for building and training CNN models for image classification."""

    def __init__(self):
        self.model = None
        self.history = None

    def create_cnn_model(self, input_shape: tuple, num_classes: int, dropout_rate: float = 0.5):
        """
        Build a Convolutional Neural Network for image classification.

        Args:
            input_shape: Tuple (height, width, channels) of input images.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.

        Returns:
            Compiled Keras Sequential model.
        """
        if tf is None:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fully connected
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, x_train, y_train, x_val=None, y_val=None,
              epochs: int = 50, batch_size: int = 32):
        """
        Train the model with early stopping and learning rate scheduling.

        Args:
            x_train: Training images array.
            y_train: Training labels (one-hot encoded).
            x_val: Validation images array.
            y_val: Validation labels (one-hot encoded).
            epochs: Maximum number of training epochs.
            batch_size: Batch size for training.

        Returns:
            Training history object.
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_cnn_model() first.")

        cb = [
            callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        validation_data = (x_val, y_val) if x_val is not None else None

        self.history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=1
        )
        return self.history

    def evaluate(self, x_test, y_test):
        """
        Evaluate the trained model on test data.

        Args:
            x_test: Test images array.
            y_test: Test labels (one-hot encoded).

        Returns:
            Dictionary with loss and accuracy.
        """
        if self.model is None:
            raise ValueError("No model available for evaluation.")

        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {"loss": round(loss, 4), "accuracy": round(accuracy, 4)}

    def predict(self, images):
        """
        Generate predictions for a batch of images.

        Args:
            images: Array of images with shape (n, height, width, channels).

        Returns:
            Array of predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("No model available for prediction.")
        return self.model.predict(images)

    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)

    def load_model(self, filepath: str):
        """Load a previously saved model from disk."""
        if tf is None:
            raise ImportError("TensorFlow is required.")
        self.model = models.load_model(filepath)
        return self.model

    def get_model_summary(self) -> str:
        """Return a string representation of the model architecture."""
        if self.model is None:
            return "No model created yet."
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)


if __name__ == "__main__":
    platform = DeepLearningPlatform()

    # Demo: build a model for CIFAR-10 style images (32x32x3, 10 classes)
    model = platform.create_cnn_model(
        input_shape=(32, 32, 3),
        num_classes=10
    )
    print("Model created successfully.")
    print(platform.get_model_summary())

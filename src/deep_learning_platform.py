#!/usr/bin/env python3
"""Deep Learning Platform"""
import tensorflow as tf
import numpy as np

class DeepLearningPlatform:
    def __init__(self):
        self.model = None
    
    def create_cnn_model(self, input_shape, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model = model
        return model

if __name__ == "__main__":
    print("Deep Learning Platform initialized")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import os
import tensorflow as tf
from sklearn.metrics import f1_score
from model import *
from sklearn.metrics import accuracy_score

main_path = "https://github.com/JesliSantiago/mlo-final-project.git/code"
data_path = os.path.join(main_path, "../data/")


def test_data(df):
    assert df['Poem'].apply(isinstance, args=(str,)).all(), "Not all poems are strings."

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    assert not df['Poem'].str.endswith(tuple(image_extensions)).any(), "Found potential image files in the Poem column."

    assert not df.isnull().any().any(), "There are missing values in the dataframe."

    valid_genres = ['Music', 'Death', 'Environment', 'Affection']
    assert df['Genre'].isin(valid_genres).all(), "Invalid genre found in the 'Genre' column."


def test_memorization(model_instance, subset_size=25, epochs=5, acc_thresh=0.95):

    original_train = model_instance.df_train.copy()

    # Set the model's df_train to the subset
    model_instance.df_train = model_instance.df_train.sample(subset_size)

    # Train the model on this small subset
    model_instance.train(epochs=epochs)

    # Evaluate the model on the same subset
    inpt = model_instance.tokenizer.texts_to_sequences(model_instance.df_train['Poem'])
    inpt = tf.keras.utils.pad_sequences(inpt, padding='pre', maxlen=model_instance.max_len)
    otpt = model_instance.le.transform(model_instance.df_train['Genre'])
    loss, acc = model_instance.model.evaluate(inpt, otpt, verbose=0)

    # Revert the model's df_train to the original data
    model_instance.df_train = original_train

    print(f"Accuracy on small subset: {acc*100:.2f}%")

    # Check if the model's accuracy on the subset meets the threshold
    assert acc >= acc_thresh, f"Model's accuracy {acc*100:.2f}% on the subset is below the expected threshold."

def plot_memorization_test(model_instance):
    if model_instance.trained_model:
        plt.figure(figsize=(12,6))
        plt.plot(model_instance.trained_model.history['acc'], label='Training Accuracy')
        try:
            plt.plot(model_instance.trained_model.history['val_acc'], label='Validation Accuracy')
        except KeyError:
            print("No validation accuracy in history. Only plotting training accuracy.")
        plt.title('Memorization Test: Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No training history available.")


def test_classification(max_epochs_per_iteration=10, max_iterations=50):  # safeguard to avoid infinite loops
    
    poem_model = poem_classifier_model()
    poem_model.load_data()
    poem_model.preprocess()

    iteration = 0
    best_accuracy = 0  # to store the best accuracy achieved
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\nTraining iteration {iteration}...")
        poem_model.train(epochs=max_epochs_per_iteration)
        
        # Predict on the validation set
        X_valid = poem_model.df_test['Poem']
        y_valid = poem_model.df_test['Genre']

        input_valid = poem_model.tokenizer.texts_to_sequences(X_valid)
        input_valid = tf.keras.utils.pad_sequences(input_valid, padding='pre', 
                                                   maxlen=poem_model.max_len)
        y_pred = poem_model.model.predict(input_valid)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_valid_transformed = poem_model.le.transform(y_valid)
        
        acc = accuracy_score(y_valid_transformed, y_pred_labels)
        
        # Store the best accuracy achieved
        if acc > best_accuracy:
            best_accuracy = acc
            poem_model.model.save("best_testclassifmodel.h5")  # saving the best model
        
        print(f"After {iteration} iterations, validation accuracy is: {acc:.2f}")
        
        # Check if the model is "good"
        poem_model._is_good()
        if acc > poem_model.thresh:
            print(f"Model met the internal threshold after {iteration} iterations.")
            break
    else:  # this part of 'else' will execute if the 'while' loop finishes without 'break'
        print(f"Max iterations reached. Best validation accuracy achieved: {best_accuracy:.2f}")

    return best_accuracy


    

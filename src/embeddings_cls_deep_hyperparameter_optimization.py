#!/usr/bin/python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, MaxPool2D
tf.config.run_functions_eagerly(True)

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import optuna
import joblib


def preprocessing(img_nrows, img_ncols):
    meta = pd.read_csv(os.path.join("..", "data", "analysis_subset", "subset_metadata.csv"))
    # Labels
    labelNames = list(set(meta['artist']))
    # Load data
    data = []
    label = []
    for i, id in enumerate(meta['ID']):
        # Preprocess and save image data
        pic_array = cv2.imread(os.path.join("..", "img", f"{id}_at_iteration_1000.png")) #load image
        compressed = cv2.resize(pic_array, (img_ncols, img_nrows), interpolation = cv2.INTER_AREA) #resize image to fit VGG-16
        data.append(compressed) 
        # Saving label
        label.append(meta['artist'][i])

    # Turn into numpy arrays (this is the format that we need for the model)
    label = np.array(label)
    data = np.array(data)

    # Normalise data
    data = data.astype("float")/255.0

    # Split data
    (trainX, testX, trainY, testY) = train_test_split(data, 
                                                        label, 
                                                        test_size=0.2)

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)  

    return trainX, testX, trainY, testY, labelNames


def create_model(input_shape, params):
    # base model
    vgg19 = tf.keras.applications.VGG19(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # transfer vgg layes into fine-tuning model
    model = Sequential()
    for layer in vgg19.layers:
        layer.trainable = False
        model.add(layer)

    model.add(Flatten())
    model.add(Dense(2**params["n_nodes_1"]))
    if params["activation_1"]:
        model.add(Activation(params['activation_1']))
    model.add(Dense(2**params["n_nodes_2"]))
    if params["activation_2"]:
        model.add(Activation(params['activation_2']))
    model.add(Dropout(0.1))
    model.add(Dense(4))
    model.add(Activation('softmax'))


    # check if layers are trainable
    num_layers = len(model.layers)
    for x in range(0, num_layers):
        print(model.layers[x].trainable)

    # compile model
    model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')

    print(model.summary())

    return model



def objective(trial):
    # Load a sample to get input dimensions
    sample_img_path = '../img/1900_the-beginning-of-life_at_iteration_1000.png'
    width, height = tf.keras.preprocessing.image.load_img(sample_img_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    channels = 3
  
    # Preprocessing
    trainX, testX, trainY, testY, labelNames = preprocessing(img_nrows, img_ncols)

    # Create model
    params = {
        "n_nodes_1": trial.suggest_int("n_nodes_1", 5, 12),
        "activation_1": trial.suggest_categorical("activation_1", [None, "relu"]),
        "activation_2": trial.suggest_categorical("activation_2", [None, "relu"])
    }
    params["n_nodes_2"] = trial.suggest_int("n_nodes_2", 5, params["n_nodes_1"])

    model = create_model(input_shape = (img_nrows, img_ncols, channels), params=params)

    # Fit model
    epochs = 1
    batch_size = 32

    H = model.fit(trainX, trainY,
            validation_data = (testX, testY),
            batch_size = batch_size,
            epochs = epochs, 
            verbose = 1
            )

    # Evaluate model
    predictions = model.predict(testX, batch_size=batch_size)

    cm = classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames,
                                output_dict=True)


    print(cm)

    return cm['accuracy']



def main():

    study = optuna.create_study(study_name="cls_deep", direction="maximize")

    study.optimize(objective, n_trials=100)

    joblib.dump(study.best_params, os.path.join("..", "models", "hyperparameter_optimization", "results.pkl"))


if __name__ == '__main__': 
    main()
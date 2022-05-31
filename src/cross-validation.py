'''
Script for classifying artists based on different versions of paintings (original or style images) using a CNN model
'''
import os
import argparse 
import pandas as pd
import numpy as np
from statistics import mean, stdev
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, MaxPool2D
tf.config.run_functions_eagerly(True)



def create_model():
    ''' Build and compile CNN model with pretrained weights
    Returns
    ----------
    model: instance of class Sequential
        Compiled model

    '''
    # Base model
    vgg19 = tf.keras.applications.VGG19(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Transfer vgg layes into fine-tuning model
    model = Sequential()
    for layer in vgg19.layers:
        layer.trainable = False
        model.add(layer)

    # Add fully connected layers
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dense(2048))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # Check if layers are trainable
    num_layers = len(model.layers)
    for x in range(0, num_layers):
        print(model.layers[x].trainable)

    # Compile model
    model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')

    return model


def main(output_name, image_type):
    '''
    Apply CNN model to the input data
    Parameters
    ----------
    output_name: str
        Filename for results to be saved under
    image_type: str
        Type of input image (specific for this project). Can be either "style" or "original"
    '''
    print("[INFO] Preparing data")

    meta = pd.read_csv(os.path.join("..", "data", "analysis_subset", "subset_metadata.csv"))

    meta = meta[meta['artist'] != "KUPKA, Frantisek"]

    # Labels
    labelNames = list(set(meta['artist']))

    # Load data
    data = []
    label = []
    for i, id in enumerate(meta['ID']):
        # Preprocess and save image data
        if image_type == "style":
            pic_array = cv2.imread(os.path.join("..", "img", f"{id}_at_iteration_1000.png")) #load style image
        if image_type == "original":
            pic_array = cv2.imread(os.path.join("..", "data", "analysis_subset", "img", f"{id}.jpg")) #load original image
        compressed = cv2.resize(pic_array, (224, 224), interpolation = cv2.INTER_AREA) #resize image to fit VGG-16
        data.append(compressed) 
        # Saving label
        label.append(meta['artist'][i])

    # Setting up for cross-validation
    epochs = 5
    batch_size = 32

    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=73)
    skf.get_n_splits(data, label)

    predictions = []
    history = []
    accuracy = []
    cm = []
    for index, (train_index, test_index) in enumerate(skf.split(data, label)):
        print(f"[INFO] Starting fold {index}")

        # New model
        model = None
        model = create_model()

        # Turn into numpy arrays (this is the format that we need for the model)
        y = label = np.array(label)
        X = data = np.array(data)

        # Normalise data
        X_norm = X.astype("float")/255.0

        # Convert labels to one-hot encoding
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y)  

        # Create splits
        x_train_fold, x_test_fold = X_norm[train_index], X_norm[test_index]
        y_train_fold, y_test_fold = y_bin[train_index], y_bin[test_index]

        # Fit model
        H = model.fit(x_train_fold, y_train_fold,
                validation_data = (x_test_fold, y_test_fold),
                batch_size = batch_size,
                epochs = epochs, 
                verbose = 1
                )

        # Save predictions, history, and classification report
        predictions_c = model.predict(x_test_fold, batch_size=batch_size)
        predictions.append(predictions_c)
        history.append(H.history)
        accuracy.append(H.history['accuracy'][4])
        cm.append(classification_report(y_test_fold.argmax(axis=1),
                                predictions_c.argmax(axis=1),
                                target_names=labelNames))

    # Saving results
    df = pd.DataFrame({
        'predictions': predictions,
        'history': history,
        'accuracy': accuracy,
        'cm': cm})
    df.to_csv(os.path.join("..", "models", f"cv_{output_name}.csv"), index=False)

    print(f"[INFO] Max accuracy: {max(accuracy)}")
    print(f"[INFO] Min accuracy: {min(accuracy)}")
    print(f"[INFO] Mean accuracy: {mean(accuracy)}")
    print(f"[INFO] SD accuracy: {stdev(accuracy)}")

if __name__ == '__main__': 
    ap = argparse.ArgumentParser(description="[INFO]")
    
    # Argument for specifying path to input folder
    ap.add_argument("-it", 
                "--image_type", 
                required=True, 
                type=str,
                choices=["style", "original"],
                help="type of images to use in classification (style or original images)") 

    ap.add_argument("-o",
                    "--output_name",
                    required=True,
                    type=str,
                    help="name/id for outputs")

    args = vars(ap.parse_args())

    # Running script
    main(output_name=args['output_name'], image_type=args['image_type'])




#python3 cross-validation.py -o "nn1_epoch5_batchsize32_s224_nokupka" -it "style"

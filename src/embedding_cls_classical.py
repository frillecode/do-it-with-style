# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from wasabi import msg

# preprocessing
import cv2
import tensorflow as tf # TODO replace dependency, don't need whole tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

# %%
# load a sample to get input dimensions
sample_img_path = '../img/1900_the-beginning-of-life_at_iteration_1000.png'
# TODO get width and height in a less crazy way
width, height = tf.keras.preprocessing.image.load_img(sample_img_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
channels = 3

# %%
meta = pd.read_csv(os.path.join("..", "data", "analysis_subset", "subset_metadata.csv"))

# Labels
labelNames = list(set(meta['artist']))

msg.info('preprocessing')
# Load data
data = []
label = []
for i, id in tqdm(enumerate(meta['ID'])):
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



# %% 
# model
msg.info('training models')
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

models = []
results = []
for name, clf in tqdm(zip(names, classifiers)):
    clf.fit(trainX, trainY)
    models.append(clf)

    predictions = clf.predict(testX)
    report = classification_report(testY.argmax(axis=1),
                        predictions.argmax(axis=1),
                        target_names=labelNames)

    results.append('\n')
    results.append(name)
    results.append(report)
    results.append('\n')


results_str = '\n'.join(results)

with open("evaluation_classical.txt", "w", encoding="utf-8") as file:
    file.write(results_str)

# %%

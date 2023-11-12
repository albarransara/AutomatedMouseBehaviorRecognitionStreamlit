# profilying memory python, %memit %mprune


import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from backboneResNet import FeatureExtractor

# Import the NN model
# model_g = tf.keras.models.load_model('resnet_lstm_accuracy_grooming.h5')
# model_mr = tf.keras.models.load_model('resnet_lstm_accuracy_mid_rearing.h5')
# model_wm = tf.keras.models.load_model('resnet_lstm_accuracy_wall_rearing.h5')

def classify_video(csv, video_name, path_to_video, behaviours):
    # Process video and csv to get the model input
    inputs = preprocess_video(csv, video_name, path_to_video)
    # Pass the input to the backbone, so we can extract the features and process them through the second model
    result_percentage, results = predict_video(inputs, behaviours)
    return result_percentage, results


# Preprocess data from the video, so we can generate the NN's input
# Split video frames and crop by the midbody position
def preprocess_video(df, video_name, path_to_video):
    expansion = 80  # Define the size of the crop

    # From the csv, get the rat's position, so we can crop the frames
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df.set_index(df.columns[0], inplace=True)

    midbody = np.concatenate((df['midbody_y'].values[:, np.newaxis], df['midbody_x'].values[:, np.newaxis]), axis=1)
    midbody = midbody.astype(int)

    # Get video and crop its frames
    vidcap = cv2.VideoCapture(path_to_video + video_name)
    success, image = vidcap.read()
    count = 0
    frames = []

    while success:
        # Crop image based on mouse midbody (make boxes having the mouse in the middle, should be square)
        top = max(0, midbody[count][0] - expansion) - max(0, midbody[count][0] + expansion - image.shape[0])
        bottom = min(image.shape[0], midbody[count][0] + expansion) + max(0, expansion - midbody[count][0])
        left = max(0, midbody[count][1] - expansion) - max(0, midbody[count][1] + expansion - image.shape[1])
        right = min(image.shape[1], midbody[count][1] + expansion) + max(0, expansion - midbody[count][1])

        frame = image[top:bottom, left:right]

        # Save frame
        frames.append(frame)

        success, image = vidcap.read()
        count += 1

    return frames


# Extract features through the Backbone model
def feature_extraction(frames):
    features = []
    feature_extractor = FeatureExtractor('resnet')

    for frame in frames:
        features.append(feature_extractor(frame))

    return np.array(features)


# Extract features through the Backbone model and get the final prediction through the final model
def predict_video(features, behaviours):
    # DFs where we will store the results for the behaviors
    results_percentage = pd.DataFrame()

    # Lists where the results will be stored
    grooming = []
    mid_rearing = []
    wall_rearing = []

    for i in range(0, len(features), 300):
        print('Batch ', i)

        f = feature_extraction(features[i: i + 300])
        f2 = np.zeros((1, 1650, 2048))
        f2[0] = f

        # Get the result for each wanted behaviour
        if "grooming" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_grooming.h5')
            grooming.append(model.predict(f2))
            print(np.array(grooming).shape)
        elif "mid_rearing" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_mid_rearing.h5')
            mid_rearing.append(model.predict(f2))
        elif "wall_rearing" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_wall_rearing.h5')
            wall_rearing.append(model.predict(f2))

    # Generate dataframes with the results
    if "grooming" in behaviours:
        results_percentage['grooming'] = list(np.concatenate(grooming[:]).flat)
    elif "mid_rearing" in behaviours:
        results_percentage['mid_rearing'] = list(np.concatenate(mid_rearing[:][0]).flat)
    elif "wall_rearing" in behaviours:
        results_percentage['wall_rearing'] = list(np.concatenate(wall_rearing[:][0]).flat)

    results = results_percentage.copy()
    results[results >= 0.5] = 1
    results[results < 0.5] = 0

    return results_percentage, results

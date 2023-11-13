# profilying memory python, %memit %mprune
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from backboneResNet import FeatureExtractor

def classify_video(csv, video_name, path_to_video, behaviours):
    # Process video and csv to get the model input
    inputs = preprocess_video(csv, video_name, path_to_video)
    # Pass the input to the backbone, so we can extract the features and process them through the second model
    results = predict_video(inputs, behaviours)
    return results

# Function to obtain mid-body coordinates of one mouse, given an image
def get_coordinates(image):
    # We can start by converting the image to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # We can rescale the values to highlight constrasted areas
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Crop unwanted part
    crop = thresh[:, :550]
    # Find bounding box and centroid
    cnts = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # We want to void noise, for that we will just consider the biggest countour since it is going to be the mouse
    c = 0
    c_len = len(cnts[0])
    for i in range(len(cnts[1:])):
        if len(cnts[i]) > c_len:
            c = i
            c_len = len(cnts[i])
    # Find centroid
    M = cv2.moments(cnts[c])
    cX = int(M["m10"] / (M["m00"] + 1e-10))
    cY = int(M["m01"] / (M["m00"] + 1e-10))
    return np.array([cX, cY])

# Preprocess data from the video, so we can generate the NN's input
# Split video frames and crop by the midbody position
def preprocess_video(df, video_name, path_to_video):
    expansion = 80  # Define the size of the crop

    # From the csv, get the rat's position, so we can crop the frames
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df.set_index(df.columns[0], inplace=True)

    # Get video and crop its frames
    vidcap = cv2.VideoCapture(path_to_video + video_name)
    success, image = vidcap.read()
    count = 0
    frames = []

    if ('midbody_y' not in df.columns) or ('midbody_x' not in df.columns):
        midbody = []
    else:
        midbody = np.concatenate((df['midbody_y'].values[:, np.newaxis], df['midbody_x'].values[:, np.newaxis]), axis=1)
        midbody = midbody.astype(int)

    while success:
        if len(midbody) == 0:
            coordinate = get_coordinates(image)
            # Crop image based on mouse midbody (make boxes having the mouse in the middle, should be square)
            top = max(0, coordinate[0] - expansion) - max(0, coordinate[0] + expansion - image.shape[0])
            bottom = min(image.shape[0], coordinate[0] + expansion) + max(0, expansion - coordinate[0])
            left = max(0, coordinate[1] - expansion) - max(0, coordinate[1] + expansion - image.shape[1])
            right = min(image.shape[1], coordinate[1] + expansion) + max(0, expansion - coordinate[1])

        else:
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
        frame = tf.convert_to_tensor(frame)
        #frame = tf.image.decode_image(frame, channels=3)
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        features.append(feature_extractor(frame))

    return np.array(features)

# Extract features through the Backbone model and get the final prediction through the final model
def predict_video(features, behaviours):
    # DFs where we will store the results for the behaviors
    results = pd.DataFrame()

    # Lists where the results will be stored
    grooming = []
    mid_rearing = []
    wall_rearing = []

    for i in range(0, len(features), 1650):
        print('Batch ', i)

        f = feature_extraction(features[i: i + 1650])
        f2 = np.zeros((1, 1650, 2048))
        f2[0] = f

        # Get the result for each wanted behaviour
        if "grooming" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_grooming.h5')
            grooming.append(model.predict(f2))
        if "mid_rearing" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_mid_rearing.h5')
            mid_rearing.append(model.predict(f2))
        if "wall_rearing" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_wall_rearing.h5')
            wall_rearing.append(model.predict(f2))

    # Generate dataframes with the results
    if "grooming" in behaviours:
        results['grooming'] = list(np.concatenate(grooming[:]).flat)
    if "mid_rearing" in behaviours:
        results['mid_rearing'] = list(np.concatenate(mid_rearing[:]).flat)
    if "wall_rearing" in behaviours:
        results['wall_rearing'] = list(np.concatenate(wall_rearing[:]).flat)
    return results
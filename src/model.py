# profilying memory python, %memit %mprune
import pandas as pd
import numpy as np
import tensorflow as tf
from backboneResNet import FeatureExtractor

# Extract features through the Backbone model
def feature_extraction(frames):
    features = []
    feature_extractor = FeatureExtractor('resnet')

    for frame in frames:
        frame = tf.convert_to_tensor(frame)
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        features.append(feature_extractor(frame))

    return np.array(features)

# Extract features through the Backbone model and get the final prediction through the final model
def predict_video(features, behaviours):
    # DFs where we will store the results for the behaviors
    results = pd.DataFrame()

    # Lists where the results will be stored
    grooming = []
    rearing = []

    # Start processing frames by batches
    for i in range(0, len(features), 300):
        print('Batch ', i)

        f = feature_extraction(features[i: i + 300])[None]

        # Get the result for each wanted behaviour
        if "Grooming" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_grooming.h5')
            grooming.append(model.predict(f))
        if "Rearing" in behaviours:
            # Load grooming model
            model = tf.keras.models.load_model('resnet_lstm_accuracy_rearing.h5')
            rearing.append(model.predict(f))

    # Generate dataframes with the results
    if "Grooming" in behaviours:
        results['Grooming'] = list(np.concatenate(grooming[:]).flat)
    if "Rearing" in behaviours:
        results['Rearing'] = list(np.concatenate(rearing[:]).flat)
    return results
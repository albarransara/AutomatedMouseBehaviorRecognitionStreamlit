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
    rearing = []

    # Start processing data by batches
    for i in range(0, len(features), 300):
        print('Batch ', i)

        f = feature_extraction(features[i: i + 300])
        f2 = np.zeros((1, 300, 2048))
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
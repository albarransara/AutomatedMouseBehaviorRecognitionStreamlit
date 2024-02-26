''' This file contains all the required methods to process a video'''
import pandas as pd
import numpy as np
import cv2

# The following method process a given frame, it crops it so only the mouse is in the center
# It will return a cropped image and the coordinates of the mouse's midbody
def crop_image(image, expansion):
    # Start by converting the image to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Rescale the values to highlight constrasted areas
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Crop the box
    thresh2 = np.invert(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] #TODO revisar perque fem aixo

    # We want to avoid noise, for that we will just consider the biggest contour since it is going to be the box
    c = 0
    c_len = len(cnts[0])
    for i in range(len(cnts[1:])):
        if len(cnts[i]) > c_len:
            c = i
            c_len = len(cnts[i])

    # Obtain bounding rectangle to get the box coordinates
    x, y, w, h = cv2.boundingRect(cnts[c])
    # If the box is way too big, we won't need to crop it, else we will
    # If the box isnot detected by the filter, then we can consider it is big enough to not crop it
    if not (x+w - x) < image.shape[0]//4:
        image = image[y:y+h, x:x+w]
        thresh = thresh[y:y+h, x:x+w]

    # We can now look for the mouse contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] #TODO revisar perque fem aixo

    # We want to avoid noise, for that we will just consider the biggest countour since it is going to be the mouse, as long as it is not
    # to close to the extremes of the image, since that would mean it is an external object
    c = 0
    c_len = len(cnts[0])
    for i in range(len(cnts[1:])):
        if len(cnts[i]) > c_len:
            # Obtain bounding rectangle to get measurements
            x, y, w, h = cv2.boundingRect(cnts[i])
            if not (y < 100) or not (y + h > image.shape[0] - 100):
                c = i
                c_len = len(cnts[i])

    # Crate a bounding box arround the mouse
    x, y, w, h = cv2.boundingRect(cnts[c])

    # Find centroid, this will be the center position of the mouse
    M = cv2.moments(cnts[c])
    cX = int(M["m10"] / (M["m00"] + 1e-10))
    cY = int(M["m01"] / (M["m00"] + 1e-10))
    midbody = [cX, cY]
    top = max(0, midbody[0] - expansion) - max(0, midbody[0] + expansion - image.shape[0])
    bottom = min(image.shape[0], midbody[0] + expansion) + max(0, expansion - midbody[0])
    left = max(0, midbody[1] - expansion) - max(0, midbody[1] + expansion - image.shape[1])
    right = min(image.shape[1], midbody[1] + expansion) + max(0, expansion - midbody[1])

    return image[left:right, top:bottom], midbody

# This function returns a cropped video as well as a list of the mouse's positions at each frame
# Given an FPS rate, it will adjust the video to it
def pos_video(video_name, path_to_video, expansion=80, fps=None):
    # Load the given video
    vidcap = cv2.VideoCapture(path_to_video + video_name)
    # Get the video's FPS rate
    fps_in = vidcap.get(cv2.CAP_PROP_FPS)

    if not vidcap.isOpened():
        raise Exception("A problem occurred when processing the video. Video couldn't be loaded.")

    # Start processing each frame
    success,image = vidcap.read()

    # Initialize all variables we willl use to store information
    frames_in = 0
    frames_out = 0
    frames = []
    pos_x = []
    pos_y = []

    while success:
        # If the video already has the standard FPS, we don't have to do anything
        if (fps_in == fps) or (fps is None):
            # Crop image so mouse is positioned in the center
            frame, coordinates = crop_image(image, expansion)
            # Save frame and position
            frames.append(frame)
            pos_x.append(coordinates[0])
            pos_y.append(coordinates[1])
            success, image = vidcap.read()

        # Else we will adjust the frames we process, so we get the specified FPS rate
        else:
            # We will calculate the second where we are on the video and scale it to the desired FPS
            out_due = int(frames_in / fps_in * fps)
            if out_due > frames_out:
                frames_out += 1
                # Crop image so mouse is positioned in the center
                frame, coordinates = crop_image(image, expansion)
                frames.append(frame)
                pos_x.append(coordinates[0])
                pos_y.append(coordinates[1])
            success, image = vidcap.read()
            frames_in += 1

    # We will create a dataframe with the mouse positions
    positions = pd.DataFrame()
    positions['midbody x'] = pos_x
    positions['midbody y'] = pos_y
    return frames, positions

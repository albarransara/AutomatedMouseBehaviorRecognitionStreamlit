import numpy as np
import cv2
from distances import *
from itertools import chain

# Method to process DCL csv with labels
def analyze_df_labeled(df, behaviours):
    # Get action at each frame
    actions = ['no_action']*df.shape[0]

    # Get columns with labels
    # First we will define which possible names the columns that correspond to labels might have
    grooming = ['grooming', 'g']
    rearing = ['rearing mig', 'rearing paret', 'mid rearing', 'wall rearing', 'rearing', 'r', 'mr', 'wr']

    # Then we will convert df indexes lower case
    df.columns = map(str.lower, df.columns)

    g_indx, r_indx = [],[]
    # Now we will check which columns correspond to results and change its names to a standard one
    # If no column matches to the expected labels, give an error message
    if 'Grooming' in behaviours:
        g_indx = np.argwhere(np.isin(df.columns, grooming)).flatten()
        if len(g_indx) <= 0:
            raise Exception("Grooming doesn't have a propperly named column on the provided csv."
                            "Possible name, they can have capital letters, are: 'rearing mig', 'rearing paret', 'mid rearing', 'wall rearing', 'rearing', 'r', 'mr', 'wr'")
        df.columns.values[g_indx] = "Grooming"
        df.Grooming[df.Grooming < 0.5] = 0

    if 'Rearing' in behaviours:
        r_indx = np.argwhere(np.isin(df.columns, rearing)).flatten()
        if len(r_indx) <= 0:
            raise Exception("Rearing doesn't have a propperly named column on the provided csv."
                            "Possible name, they can have capital letters, are: 'grooming', 'g'")
        df.columns.values[r_indx] = "Rearing"
        df.Rearing[df.Rearing < 0.5] = 0

    # Now that we have the indexes we will extract the labels for the video tagging
    labels = df.iloc[:,list(chain.from_iterable([g_indx, r_indx]))]
    for ind, row in labels.iterrows():
        if max(row) != 0:
            b_max = np.argmax(row)
            actions[ind] = labels.columns[b_max]

    # Now caculate cumulative distances
    try:
        distance_frame = distances_DLC(df)
    except:
        raise Exception('Distances can not be properly calculated.')

    # Return processed results and new df containing distance data
    return (actions, distance_frame)

# Function to annotate each video frame
def annotate_video(labels,video_name,path_to_video, fps=None):
    # Position of the annotations
    position = (10,50)
    # Open the video
    cap = cv2.VideoCapture(path_to_video+video_name)

    if not cap.isOpened():
        raise Exception("A problem occurred when tagging the video. Video couldn't be loaded.")

    # Get data from the video
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter("out_"+video_name,cv2.VideoWriter_fourcc(*'mp4v'),framespersecond,(frame_width,frame_height))
    returned, frame = cap.read()
    frame_count = 0
    frames_out = 0
    label_out = labels[frames_out]

    # Iterate through frames
    while returned:
        if (fps == framespersecond) or (fps is None):
            label_out = labels[frame_count] if labels[frame_count] != "no_action" else "No action"
            cv2.putText(frame,label_out,position,cv2.FONT_HERSHEY_SIMPLEX,1,(209,80,0,255),3)
            out.write(frame)
            frame_count += 1
            returned, frame = cap.read()
        else:
            # We will calculate the second where we are on the video and scale it to the desired FPS
            out_due = int(frame_count / framespersecond * fps)
            if out_due > frames_out:
                label_out = labels[frames_out] if labels[frames_out] != "no_action" else "No action"
                frames_out += 1
            cv2.putText(frame, label_out, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
            out.write(frame)
            returned, frame = cap.read()
            frame_count += 1

    cap.release()
    out.release()

    return framespersecond

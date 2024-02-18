from statistics import mean, median
import math
import pandas as pd
import numpy as np
import cv2
from distances import *
from itertools import chain

class Period:
    def __init__(self,start:int,action=str):
        self.start = start
        self.end = None
        ## should either be "grooming", "rearing_mid", or "rearing_wall"
        self.action = action

    def set_end(self,end:int):
        self.end = end

    def get_length(self):
        if self.end is None:
            return 0
        else:
            return self.end-self.start + 1

    def __repr__(self):
        return f"<Period {self.action}: {self.start}-{self.end}>"

    def __lt__(self,other):
        return self.end < other.end

def get_current_action(line):
    grooming, rearing_mid, rearing_wall = line[4:7]

    if grooming == 1:
        return "grooming"
    elif rearing_mid == 1:
        return "rearing_mid"
    elif rearing_wall == 1:
        return "rearing_wall"
    else:
        return "no_action"

## get mean, median
def get_summary_data(period_set):
    length_list = []
    for period in period_set:
        length_list.append(period.get_length())

    if len(length_list) == 0:
        return (0,0)
    
    mean_length = mean(length_list)
    median_length = median(length_list)

    return (mean_length,median_length)

# TODO delate this
def get_dist(xdiff,ydiff):
    return math.sqrt(xdiff**2+ydiff**2)

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
            return -1
        df.columns.values[g_indx] = "Grooming"
    if 'Rearing' in behaviours:
        r_indx = np.argwhere(np.isin(df.columns, rearing)).flatten()
        if len(r_indx) <= 0:
            return -1
        df.columns.values[r_indx] = "Rearing"

    # Now that we have the indexes we will extract the labels for the video tagging
    labels = df.iloc[:,list(chain.from_iterable([g_indx, r_indx]))]
    for ind, row in labels.iterrows():
        if max(row) != 0:
            b_max = np.argmax(row)
            actions[ind] = labels.columns[b_max]

    # Now caculate cumulative distances
    distance_frame = distances_DLC(df)

    # TODO mirar de borrar esto
    '''
    df['actions'] = actions
        if 'grooming' in results.columns:
        results.loc[results['grooming'] < threshold,'grooming'] = 0
    if 'mid_rearing' in results.columns:
        results.loc[results['mid_rearing'] < threshold,'mid_rearing'] = 0
    if 'wall_rearing' in results.columns:
        results.loc[results['wall_rearing'] < threshold,'wall_rearing'] = 0
    print(results.head())
    # Then apply argmax
    for ind,row in results.iterrows():
        if max(row) != 0:
            b_max = np.argmax(row)
            actions[ind] = results.columns[b_max]
            results[ind,b_max] = 1

    df['actions'] = actions
    # We can clean results
    results[results != 1] = 0
    '''
    #TODO mirar de borrar esto
    '''
    #Calculate how much distance the mouse has moved
    d_x = []
    d_y = []
    d_t = []
    cd_x = []
    cd_y = []
    cd_t = []

    df_slice = df.iloc[:,-3:-1]
    for ind, row in df_slice.iterrows():
        if ind < df.shape[0]-1:
            x_diff = abs(float(row[0])-float(df_slice[df_slice.columns[0]][ind+1]))
            y_diff = abs(float(row[1])-float(df_slice[df_slice.columns[1]][ind+1]))
            t_diff = get_dist(x_diff,y_diff)

            d_x.append(x_diff)
            d_y.append(y_diff)
            d_t.append(t_diff)

            if ind == 0:
                cd_x.append(x_diff)
                cd_y.append(y_diff)
                cd_t.append(t_diff)
            else:
                cd_x.append(x_diff+cd_x[-1])
                cd_y.append(y_diff+cd_y[-1])
                cd_t.append(t_diff+cd_t[-1])

    distance_frame = pd.DataFrame()
    distance_frame['frames'] = [x for x in range(1,num_frames)]
    distance_frame['d_x'] = d_x
    distance_frame['d_y'] = d_y
    distance_frame['d_t'] = d_t
    distance_frame['cd_x'] = cd_x
    distance_frame['cd_y'] = cd_y
    distance_frame['cd_t'] = cd_t
    '''

    # Return processed results and new df containing distance data
    return (actions, distance_frame)

# Method to post process results
def analyze_df(df,results):
    num_frames = df.shape[0]
    # Get action at each frame
    actions = ['no_action']*num_frames

    # First filter data with threshold
    threshold = 0.8
    if 'grooming' in results.columns:
        results.loc[results['grooming'] < threshold,'grooming'] = 0
    if 'mid_rearing' in results.columns:
        results.loc[results['mid_rearing'] < threshold,'mid_rearing'] = 0
    if 'wall_rearing' in results.columns:
        results.loc[results['wall_rearing'] < threshold,'wall_rearing'] = 0
    print(results.head())
    # Then apply argmax
    for ind,row in results.iterrows():
        if max(row) != 0:
            b_max = np.argmax(row)
            actions[ind] = results.columns[b_max]
            results[ind,b_max] = 1

    df['actions'] = actions
    # We can clean results
    results[results != 1] = 0

    #TODO revisar esto
    #Calculate how much distance the mouse has moved
    d_x = []
    d_y = []
    d_t = []
    cd_x = []
    cd_y = []
    cd_t = []

    df_slice = df.iloc[:,-3:-1]
    for ind, row in df_slice.iterrows():
        if ind < num_frames-1:
            x_diff = abs(float(row[0])-float(df_slice[df_slice.columns[0]][ind+1]))
            y_diff = abs(float(row[1])-float(df_slice[df_slice.columns[1]][ind+1]))
            t_diff = get_dist(x_diff,y_diff)

            d_x.append(x_diff)
            d_y.append(y_diff)
            d_t.append(t_diff)

            if ind == 0:
                cd_x.append(x_diff)
                cd_y.append(y_diff)
                cd_t.append(t_diff)
            else:
                cd_x.append(x_diff+cd_x[-1])
                cd_y.append(y_diff+cd_y[-1])
                cd_t.append(t_diff+cd_t[-1])

    distance_frame = pd.DataFrame()
    distance_frame['frames'] = [x for x in range(1,num_frames)]
    distance_frame['d_x'] = d_x
    distance_frame['d_y'] = d_y
    distance_frame['d_t'] = d_t
    distance_frame['cd_x'] = cd_x
    distance_frame['cd_y'] = cd_y
    distance_frame['cd_t'] = cd_t

    # Return processed results and new df containing distance data
    return (actions, distance_frame,results)

# Function to annotate each video frame
def annotate_video(labels,video_name,path_to_video):
    # Position of the annotations
    position = (10,50)
    # Open the video
    cap = cv2.VideoCapture(path_to_video+video_name)

    if not cap.isOpened():
        print("Unable to read camera feed")
        # TODO return error

    # Get data from the video
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter("out_"+video_name,cv2.VideoWriter_fourcc(*'mp4v'),10,(frame_width,frame_height))
    returned = True
    frame_count = 0

    # Iterate through frames
    while returned:
        returned, frame = cap.read()
        if returned:
            label_out = labels[frame_count] if labels[frame_count] != "no_action" else "No action"
            cv2.putText(frame,label_out,position,cv2.FONT_HERSHEY_SIMPLEX,1,(209,80,0,255),3)
            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()

    return framespersecond

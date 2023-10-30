from statistics import mean, median
import math
import pandas as pd
import numpy as np
import cv2

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
    # print(grooming,rearing_mid,rearing_wall)

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

def get_dist(xdiff,ydiff):
    return math.sqrt(xdiff**2+ydiff**2)

# TODO arreclar aquest metode
def analyze_df(df,results):

    #label_names = df.columns[1:-2] # Get the possible actions names, ESTA MAL, pilla columnas de la psoicion del animal

    num_frames = df.shape[0]

    # Get action at each frame
    actions = ['no_action']*num_frames
    
    # First filter data with threshold
    thershold = 0.8 
    results.loc[results['grooming'] < thershold,'grooming'] = 0
    results.loc[results['mid_rearing'] < thershold,'mid_rearing'] = 0
    results.loc[results['wall_rearing'] < thershold,'wall_rearing'] = 0
    
    # Then apply argmax
    for ind,row in results.iterrows():
        if max(row) != 0:
            actions[ind] = results.columns[np.argmax(row)]

    df['actions'] = actions

    d_x = []
    d_y = []
    d_t = []
    cd_x = []
    cd_y = []
    cd_t = []


    #TODO revisar esto
    #Calculate how much distance the mouse has moved
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

    # Return [frames,actions] of original df, and new df containing distance data
    return (df[[df.columns[0],df.columns[-1]]], distance_frame)



def annotate_video(labels,video_name,path_to_video):

    position = (10,50)
    cap = cv2.VideoCapture(path_to_video+video_name)
    print(path_to_video+video_name)

    if not cap.isOpened():
        print("Unable to read camera feed")

    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter("out_"+video_name,cv2.VideoWriter_fourcc(*'mp4v'),10,(frame_width,frame_height))
    returned = True
    frame_count = 0

    print(labels.shape)
    print(labels)

    while returned:
        returned, frame = cap.read()
        if returned:
            label_out = labels['actions'].iloc[frame_count] if labels['actions'].iloc[frame_count] != "no_action" else "0"
            # print(label_out)
            cv2.putText(frame,label_out,position,cv2.FONT_HERSHEY_SIMPLEX,1,(209,80,0,255),3)
            out.write(frame)
            frame_count += 1
            print(frame_count)

    cap.release()
    out.release()
    print("finished")

    return framespersecond

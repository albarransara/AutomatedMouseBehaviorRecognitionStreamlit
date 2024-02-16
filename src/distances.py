"""This file contains methods to calculate the distance the distances"""
import pandas as pd


# This function will calculate distances from a DeepLabCuts csv files
def distances_DLC(df):
    cd_x = [] # Array to caclulate the cumulative horizontal distance
    cd_y = [] # Array to calculate the cumulative vertical distance
    cd_total = [] # Array to calculate the total cumulative  distance

    # For that we will look at midbody x', 'midbody y' columns
    positions = df[['midbody x', 'midbody y']]
    
    # We will iterate through rows to caclulate distances
    for ind, row in positions.iterrows():
        # For the first position we won't consider any movement
        if ind == 0:
            cd_x.append(0)
            cd_y.append(0)
            cd_total.append(0)
        else:
            # Get the frame distance
            x_dist = abs(positions['midbody x'][ind] - positions['midbody x'][ind-1])
            y_dist = abs(positions['midbody y'][ind] - positions['midbody y'][ind-1])
            total_dist = x_dist+y_dist
            # Add new cumulative distance
            cd_x.append(cd_x[-1] + x_dist)
            cd_y.append(cd_y[-1] + y_dist)
            cd_total.append(cd_total[-1] + total_dist)

    # Finally generate a df with the distances and return it
    distance_frame = pd.DataFrame()
    distance_frame['frames'] = [x for x in range(1,df.shape[0]+1)]
    distance_frame['x'] = cd_x
    distance_frame['y'] = cd_y
    distance_frame['total'] = cd_total

    return distance_frame





import streamlit as st
import pandas as pd
from helpers import *
from model import *
import zipfile
import itertools


#os.system('pip install -r requirements.txt')

# Define applications title
st.title("Automated Mouse Behavior Recognition")


def write_bytesio_to_file(filename,bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
    # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())


@st.cache_data
def analyze_files(labels,video_name):
    return annotate_video(labels,video_name,"")


mode = st.tabs(["Manual","Automatic"])
zip_name = "results"

# Second page
with mode[1]:

    st.title("Automatic Mode")
    st.text("In Automatic Mode you can upload as many videos as you want, as long as\nyou also upload their corresponding DeepLabCut csv files.")

    #model = st.radio("Choose model",("resnet.LSTM","inception_resnet.LSTM","resnet.TCN","inception_resnet.TCN"))
    st.text("You can choose between the behaviours you want to consider: ")
    st.checkbox("Grooming", key='grooming_auto')
    st.checkbox("Mid rearing ", key='mid_rearing_auto')
    st.checkbox("Wall rearing ", key='wall_rearing_auto')

    st.title("\n")

    uploaded_videos = st.file_uploader("Upload Video files",type=["mp4"],accept_multiple_files=True,key="automatic")

    try:
        with open(f"{zip_name}.zip","rb") as fp:
            btn = st.download_button(label="Download results", key='download_auto', data=fp,file_name=f"{zip_name}.zip",mime="application/zip")
    except IOError:
            btn = st.button(label="Download results",disabled=True)

# First page
with (mode[0]):

    st.title("Manual Mode")
    st.text("In Manual Mode you can upload videos with their corresponding DeepLabCut csv file.")

    st.text("You can choose between the behaviours you want to consider: ")
    grooming_manual = st.checkbox("Grooming", key='grooming_manual')
    mid_rearing_manual = st.checkbox("Mid rearing ", key='mid_rearing_manual')
    wall_rearing_manual = st.checkbox("Wall rearing ", key='wall_rearing_manual')

    st.title("\n")

    uploaded_csvs = st.file_uploader("Upload CSV files",type=["csv"],accept_multiple_files=True)
    uploaded_videos = st.file_uploader("Upload Video files",type=["mp4"],accept_multiple_files=True)

    with st.sidebar:
        time_unit = st.radio("Choose display unit",("seconds","frames"))

    video_names = set() # Variable where we will save all the uploaded videos

    for uploaded_video in uploaded_videos:
        write_bytesio_to_file(uploaded_video.name,uploaded_video) # Add video to video buffer in bytes
        video_names.add(uploaded_video.name[:-4]+".mp4") # Add video's names to a set of all names


    if len(uploaded_csvs) > 0:
        tab_names = [] # List of csv files names

        for ind, uploaded_csv in enumerate(uploaded_csvs):
            tab_names.append(uploaded_csv.name[:-4])

        does_match = True

        # Check if for each uploaded video there is a corresponding csv
        for csv_name in tab_names:
            corresponding_video_name = csv_name.split("_")[2]+".mp4"
            if corresponding_video_name not in video_names or len(uploaded_csvs) != len(uploaded_videos):
                st.write("Make sure that each video has a corresponding .csv file and vice-versa")
                does_match = False
                break

        # If all videos have their corresponding csv's, we process them and generate the results
        if does_match:
            # Create the zip file where we will save the results
            z = zipfile.ZipFile(f"{zip_name}.zip",mode="w")

            matrix = []

            for uploaded_csv in uploaded_csvs:
                write_bytesio_to_file(uploaded_csv.name,uploaded_csv) # Add csv to video cav in bytes
                z.write(uploaded_csv.name)

            # For each uploaded video we create a tab for its results
            tabs = st.tabs(tab_names)

            # From the checkboxes, get the behaviours the user wants to predict
            behaviours = []
            if grooming_manual:
                behaviours.append("grooming")
            elif mid_rearing_manual:
                behaviours.append("mid_rearing")
            elif wall_rearing_manual:
                behaviours.append("wall_rearing")

            # Get results for each video
            for index,tab in enumerate(tabs):
                with tab:

                    # Get name of the video that will get analyzed
                    video_name = uploaded_csvs[index].name.split("_")[2][:-4] + ".mp4"
                    # Create data frame from video's csv
                    df = pd.read_csv(uploaded_csvs[index])

                    # Pass to the model the video and the csv and obtain prediction
                    result_percentage, results = classify_video(df, video_name, "", behaviours)
                    print(results)

                    #TODO
                    labels, distances = analyze_df(df,results) # TODO arreclar aquest metode,per recuento de distanci i extraccio labels
                    
                    fps = analyze_files(labels,video_name) # TODO arreclar aquest metode,per sobrescriure accions a video

                    distances['seconds'] = distances['frames'].map(lambda x: x/fps)

                    st.write("fps: ",fps)
                    z.write("out_"+video_name)

                    video_file = open("out_"+video_name, 'rb')

                    st.video(video_file)
                    
                    st.write('Horizontal distance traveled over time')
                    st.line_chart(distances[[time_unit,'d_x']],x=time_unit)

                    st.write('Vertical distance traveled over time')
                    st.line_chart(distances[[time_unit,'d_y']],x=time_unit)

                    st.write('Total distance traveled over time')
                    st.line_chart(distances[[time_unit,'d_t']],x=time_unit)

                    st.write('Cumulative horizontal distance traveled over time')
                    st.line_chart(distances[[time_unit,'cd_x']],x=time_unit)

                    st.write('Cumulative vertical distance traveled over time')
                    st.line_chart(distances[[time_unit,'cd_y']],x=time_unit)

                    st.write('Cumulative total distance traveled over time')
                    st.line_chart(distances[[time_unit,'cd_t']],x=time_unit)


            ### CREATE SUMMARY CSV HERE ###
            distances.to_csv("distance_"+video_name[:-4]+".csv",index=False)
            result_percentage.to_csv("result_percentatge_" + video_name[:-4] + ".csv", index=False)
            results.to_csv("result_binary_"+video_name[:-4]+".csv",index=False)

            z.write("distance_"+video_name[:-4]+".csv")
            z.write("result_binary_"+video_name[:-4]+".csv")
            z.write("result_percentatge_"+video_name[:-4]+".csv")

            z.close()

            with open(f"{zip_name}.zip","rb") as fp:
                btn = st.download_button(label="Download results", key='download_manual', data=fp,file_name=f"{zip_name}.zip",mime="application/zip")


            



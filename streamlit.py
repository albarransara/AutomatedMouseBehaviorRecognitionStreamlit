import streamlit as st
import pandas as pd
from helpers import *
import zipfile

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

@st.cache
def analyze_files(labels,video_name):
    return annotate_video(labels['actions'],video_name,"")


mode = st.tabs(["Manual","Automatic"])
zip_name = "results"

with mode[1]:
    st.title("Automatic Mode")
    uploaded_videos = st.file_uploader("Upload Video files",type=["mp4"],accept_multiple_files=True,key="automatic")
    model = st.radio("Choose model",("resnet.LSTM","inception_resnet.LSTM","resnet.TCN","inception_resnet.TCN"))
    try:
        with open(f"{zip_name}.zip","rb") as fp:
            btn = st.download_button(label="Download results",data=fp,file_name=f"{zip_name}.zip",mime="application/zip")
    except IOError:
            btn = st.download_button(label="Download results",data=None,disabled=True)

with mode[0]:
    st.title("Manual Mode")
    uploaded_csvs = st.file_uploader("Upload CSV files",type=["csv"],accept_multiple_files=True)
    uploaded_videos = st.file_uploader("Upload Video files",type=["mp4"],accept_multiple_files=True)

    with st.sidebar:
        time_unit = st.radio("Choose display unit",("seconds","frames"))

    video_names = set()

    for uploaded_video in uploaded_videos:
        write_bytesio_to_file(uploaded_video.name,uploaded_video)
        video_names.add(uploaded_video.name[:-4]+".mp4")


    if len(uploaded_csvs) > 0:
        tab_names = []

        for ind, uploaded_csv in enumerate(uploaded_csvs):
            tab_names.append(uploaded_csv.name[:-4])

        does_match = True
        for csv_name in tab_names:
            corresponding_video_name = csv_name.split("_")[2]+".mp4"
            if corresponding_video_name not in video_names or len(uploaded_csvs) != len(uploaded_videos):
                st.write("Make sure that each video has a corresponding .csv file and vice-versa")
                does_match = False
                break
        
        if does_match:
            z = zipfile.ZipFile(f"{zip_name}.zip",mode="w")

            matrix = []

            for uploaded_csv in uploaded_csvs:
                write_bytesio_to_file(uploaded_csv.name,uploaded_csv)
                z.write(uploaded_csv.name)
            
            tabs = st.tabs(tab_names)

            for index,tab in enumerate(tabs):
                with tab:
                    df = pd.read_csv(uploaded_csvs[index])

                    labels,distances = analyze_df(df)
                    
                    video_name = uploaded_csvs[index].name.split("_")[2][:-4] + ".mp4"
                    fps = analyze_files(labels,video_name)

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
            z.write("distance_"+video_name[:-4]+".csv")
            z.close()

            with open(f"{zip_name}.zip","rb") as fp:
                btn = st.download_button(label="Download results",data=fp,file_name=f"{zip_name}.zip",mime="application/zip")
    

            



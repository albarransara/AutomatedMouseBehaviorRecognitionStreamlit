import streamlit as st
from helpers import *
from model import *
from video_processing import *
import zipfile
import time
# os.system('pip install -r requirements.txt')

# Define applications title
st.title("Automated Mouse Behavior Recognition")

# Convert video to a bytes buffer
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

mode = st.tabs(["Manual", "Automatic", "New behaviours"])
zip_name = "results"

# First page, Manual
with mode[0]:
    st.title("Manual Mode")
    st.markdown(
        "In manual mode you can upload videos, with their corresponding DeepLabCut labeled csv, files and you will get the video with labels, as well as some stadistics.")

    st.markdown("You can choose between the behaviours you want to consider: ")
    grooming_manual = st.checkbox("Grooming", key='grooming_manual')
    rearing_manual = st.checkbox("Rearing ", key='rearing_manual')

    st.title("\n")

    uploaded_csvs_manual = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True,
                                            key="manualcsv")
    uploaded_videos_manual = st.file_uploader("Upload Video files", type=["mp4"], accept_multiple_files=True,
                                             key="manualvideos")

    st.title("\n")

    with st.sidebar:
        time_unit_manual = st.radio("Choose display unit", ("seconds", "frames"), key="manualradio")

    video_names_manual = set() # Set to save uploaded videos' names

    # Check uploaded videos
    if len(uploaded_videos_manual) > 0:
        for uploaded_video in uploaded_videos_manual:
            write_bytesio_to_file(uploaded_video.name, uploaded_video) #Convert mp4 videos to bytes
            video_names_manual.add(uploaded_video.name[:-4] + ".mp4") #Save videos' names`
    else:
        st.write("No mp4 videos have been uploaded yet")

    # Check uploaded csvs
    if len(uploaded_csvs_manual) > 0:
        tab_names_manual = []

        # Get each uploaded csv
        for ind, uploaded_csv in enumerate(uploaded_csvs_manual):
            tab_names_manual.append(uploaded_csv.name[:-4])

        # Check if they correspond to a video
        does_match_manual = True
        for csv_name in tab_names_manual:
            corresponding_video_name = csv_name.split("_")[2] + ".mp4"
            if corresponding_video_name not in video_names_manual or len(uploaded_csvs_manual) != len(
                    uploaded_videos_manual):
                st.write("Make sure that each mp4 video has a corresponding .csv file and vice-versa")
                does_match_manual = False
                break

        # In case all videos have their corresponding csv
        if does_match_manual:

            # Check if any behaviour checkbox is marked
            # From the checkboxes, get the behaviours the user wants to predict
            behaviours_manual = []
            if grooming_manual:
                behaviours_manual.append("Grooming")
            if rearing_manual:
                behaviours_manual.append("Rearing")

            if len(behaviours_manual) <= 0:
                st.write("At least one behaviour must be selected.")
            else:
                # Create a zip for the results
                z_manual = zipfile.ZipFile(f"{zip_name}.zip", mode="w")

                # TODO mirar de borrar esto
                #for uploaded_csv in uploaded_csvs_manual:
                #write_bytesio_to_file(uploaded_csv.name, uploaded_csv)
                #z_manual.write(uploaded_csv.name)

                # Open one tab for each video
                tabs = st.tabs(tab_names_manual)

                # Process each video
                for index, tab in enumerate(tabs):
                    with tab:
                        df = pd.read_csv(uploaded_csvs_manual[index])
                        # Create a progress bar so the user recives a feedback
                        progress_manual = "Operation in progress. Please wait."
                        bar_manual = st.progress(0, text=progress_manual)

                        try:
                            bar_manual.progress(40, text='Calculating distances')
                            # Get tags for the video frames and distances data
                            labels_manual, distances_manual = analyze_df_labeled(df, behaviours_manual)

                            # Get name for the new tagged video
                            bar_manual.progress(70, text='Tagging video')
                            video_name = uploaded_csvs_manual[index].name.split("_")[2][:-4] + ".mp4"
                            # Annotate the video and get its frames per second
                            fps_manual = annotate_video(labels_manual, video_name,"")

                            bar_manual.progress(90, text='Creating plots')
                            distances_manual['seconds'] = distances_manual['frames'].map(lambda x: x / fps_manual)

                            # Create a data frame to store behaviour over time
                            actions_manual = pd.DataFrame()
                            actions_manual['frames'] = distances_manual['frames']
                            actions_manual['seconds'] = distances_manual['frames']

                            st.write("fps: ", fps_manual)
                            z_manual.write("out_" + video_name)

                            video_file = open("out_" + video_name, 'rb')
                            st.video(video_file)

                            # Create one tab for behaviour graphics and one for distance ones
                            tabs_graphics = st.tabs(['Distances visualization', 'Behaviours visualization'])

                            with tabs_graphics[0]:
                                st.write('Cumulative horizontal distance traveled over time')
                                st.line_chart(distances_manual[[time_unit_manual, 'x']], x=time_unit_manual)

                                st.write('Cumulative vertical distance traveled over time')
                                st.line_chart(distances_manual[[time_unit_manual, 'y']], x=time_unit_manual)

                                st.write('Cumulative total distance traveled over time')
                                st.line_chart(distances_manual[[time_unit_manual, 'total']], x=time_unit_manual)

                            with tabs_graphics[1]:
                                if 'Grooming' in behaviours_manual:
                                    actions_manual['Grooming'] = list(map(lambda x:1 if x == 'Grooming' else 0, labels_manual))
                                    st.write('Grooming over time')
                                    st.line_chart(actions_manual[[time_unit_manual, 'Grooming']], x=time_unit_manual)

                                if 'Rearing' in behaviours_manual:
                                    actions_manual['Rearing'] = list(map(lambda x:1 if x == 'Rearing' else 0, labels_manual))
                                    st.write('Rearing over time')
                                    st.line_chart(actions_manual[[time_unit_manual, 'Rearing']], x=time_unit_manual)

                            bar_manual.progress(100, text='Video analysis has been completed!')
                            time.sleep(0.1)
                            bar_manual.empty()

                        except Exception as e:
                            st.write('An error occurred during the video processing')
                            st.write(e)
                            bar_manual.empty()

                ### CREATE SUMMARY CSV HERE ###
                #TODO mirar de borrar aquesta linea
                #distances.to_csv("distance_" + video_name[:-4] + ".csv", index=False)
                #z_manual.write("distance_" + video_name[:-4] + ".csv")
                z_manual.close()

                with open(f"{zip_name}.zip", "rb") as fp:
                    btn = st.download_button(label="Download results", data=fp, file_name=f"{zip_name}.zip",
                                             mime="application/zip")

# Second page, automatic
with (mode[1]):

    # Define all frontend components for Automatic tab
    st.title("Automatic Mode")
    st.markdown(
        "In automatic mode, you can upload videos and you will automatically get them labeled.")

    st.markdown("You can choose between the behaviours you want to consider: ")
    grooming_automatic = st.checkbox("Grooming", key='grooming_automatic')
    rearing_automatic = st.checkbox("Rearing ", key='rearing_automatic')

    st.title("\n")

    uploaded_videos = st.file_uploader("Upload Video files", type=["mp4"], accept_multiple_files=True)

    st.title("\n")

    with st.sidebar:
        time_unit = st.radio("Choose display unit", ("seconds", "frames"), key="automaticradio")

    video_names = set()  # Set to save uploaded videos' names
    does_match = False

    # Check uploaded videos
    if len(uploaded_videos) > 0:
        for uploaded_video in uploaded_videos:
            write_bytesio_to_file(uploaded_video.name, uploaded_video)  # Convert mp4 videos to bytes
            video_names.add(uploaded_video.name[:-4] + ".mp4")  # Save videos' names`
            does_match = True
    else:
        st.write("No mp4 videos have been uploaded yet")

    # If there are any videos uploaded, we can process them and generate the results
    if does_match:
        # Check if any behaviour checkbox is marked
        # From the checkboxes, get the behaviours the user wants to predict
        behaviours = []
        if grooming_automatic:
                behaviours.append("Grooming")
        if rearing_automatic:
                behaviours.append("Rearing")

        if len(behaviours) <= 0:
                st.write("At least one behaviour must be selected.")

        # If everything is in order, we can start processing the videos
        else:
            # Create a zip for the results
            z = zipfile.ZipFile(f"{zip_name}.zip", mode="w")

            # Open one tab for each video
            tabs_auto = st.tabs(video_names)
            names= list(video_names)

            # Process each video
            for index, tab in enumerate(tabs_auto):
                with tab:
                    video_name_auto = names[index]
                    # Create a progress bar so the user recives a feedback
                    progress = "Operation in progress. Please wait."
                    bar = st.progress(0, text=progress)
                    try:
                        # Preprocess the video
                        bar.progress(20, text='Preprocessing the video, getting mouse position.')
                        frames, positions = pos_video(video_name_auto, "",expansion=80, fps=10)

                        # Predict behaviours at each frame
                        bar.progress(40, text='Model predicting beheaviours at each frame.')
                        results = predict_video(frames, behaviours)

                        # Get tags for the video frames and distances data
                        bar.progress(70, text='Calculating distances')
                        labels, distances = analyze_df_labeled(pd.concat([results,positions], axis=1), behaviours)

                        # Get name for the new tagged video
                        bar.progress(80, text='Tagging video')
                        # Annotate the video and get its frames per second
                        fps = annotate_video(labels, video_name_auto, "", fps=10)

                        bar.progress(90, text='Creating plots')
                        distances['seconds'] = distances['frames'].map(lambda x: x / fps)

                        # Create a data frame to store behaviour over time
                        actions = pd.DataFrame()
                        actions['frames'] = distances['frames']
                        actions['seconds'] = distances['frames']

                        st.write("fps: ", fps)
                        z.write("out_" + video_name_auto)

                        video_file = open("out_" + video_name_auto, 'rb')
                        st.video(video_file)

                        # Create one tab for behaviour graphics and one for distance ones
                        tabs_graphics = st.tabs(['Distances visualization', 'Behaviours visualization'])

                        with tabs_graphics[0]:
                            st.write('Cumulative horizontal distance traveled over time')
                            st.line_chart(distances[[time_unit, 'x']], x=time_unit)

                            st.write('Cumulative vertical distance traveled over time')
                            st.line_chart(distances[[time_unit, 'y']], x=time_unit)

                            st.write('Cumulative total distance traveled over time')
                            st.line_chart(distances[[time_unit, 'total']], x=time_unit)

                        with tabs_graphics[1]:
                            if 'Grooming' in behaviours:
                                actions['Grooming'] = list(map(lambda x: 1 if x == 'Grooming' else 0, labels))
                                st.write('Grooming over time')
                                st.line_chart(actions[[time_unit, 'Grooming']], x=time_unit)

                            if 'Rearing' in behaviours:
                                actions['Rearing'] = list(map(lambda x: 1 if x == 'Rearing' else 0, labels))
                                st.write('Rearing over time')
                                st.line_chart(actions[[time_unit, 'Rearing']], x=time_unit)

                        bar.progress(100, text='Video analysis has been completed!')
                        time.sleep(0.1)
                        bar.empty()

                    except Exception as e:
                        st.write('An error occurred during the video processing')
                        st.write(e)
                        bar.empty()

            z.close()
            with open(f"{zip_name}.zip", "rb") as fp:
                btn = st.download_button(label="Download results", data=fp, file_name=f"{zip_name}.zip",
                                         mime="application/zip")


# Third page, automatic
with (mode[2]):
    st.write("This functionality still needs to be implemented.")
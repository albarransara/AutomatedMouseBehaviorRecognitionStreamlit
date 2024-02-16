import streamlit as st
from helpers import *
from model import *
import zipfile

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

# Perform distance analysis of the video
@st.cache_data
#TODO change directly on the code
def analyze_files(labels, video_name):
    return annotate_video(labels, video_name, "")


mode = st.tabs(["Manual", "Automatic", "New behaviours"])
zip_name = "results"

# First page, Manual
with mode[0]:
    st.title("Manual Mode")
    st.markdown(
        "In manual mode you can upload videos, with their corresponding DeepLabCut labeled csv, files and you will get the video with labels, as well as some stadistics.")

    st.markdown("You can choose between the behaviours you want to consider: ")
    grooming_manual = st.checkbox("Grooming", key='grooming_manual')
    rearing_manual = st.checkbox("Rearing ", key='mid_rearing')

    uploaded_csvs_manual = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True,
                                            key="manualcsv")
    uploaded_videos_manual = st.file_uploader("Upload Video files", type=["mp4"], accept_multiple_files=True,
                                              key="manualvideos")

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

            # Check if any behaviour cehckbox is marked
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

                matrix = [] # TODO revisar para que sirve

                # TODO mirar de borrar esto
                '''for uploaded_csv in uploaded_csvs_manual:
                #    write_bytesio_to_file(uploaded_csv.name, uploaded_csv)
                #    z_manual.write(uploaded_csv.name)
                '''
                # Open one yab for each video
                tabs = st.tabs(tab_names_manual)

                # Process each video
                for index, tab in enumerate(tabs):
                    with tab:
                        df = pd.read_csv(uploaded_csvs_manual[index])

                        try:
                            # Get tags for the video frames and distances data
                            labels, distances = analyze_df_labeled(df, behaviours_manual)
                        except:
                            st.write("The provied csv doesn't have a proper named column with the labels")
                            # Jump to the following video
                            continue

                        # Get name for the new tagged video
                        video_name = uploaded_csvs_manual[index].name.split("_")[2][:-4] + ".mp4"
                        # Annotate the video and get its frames per second
                        fps = analyze_files(labels, video_name)

                        distances['seconds'] = distances['frames'].map(lambda x: x / fps)

                        st.write("fps: ", fps)
                        z_manual.write("out_" + video_name)

                        video_file = open("out_" + video_name, 'rb')
                        st.video(video_file)

                        st.write('Cumulative horizontal distance traveled over time')
                        st.line_chart(distances[[time_unit_manual, 'x']], x=time_unit_manual)

                        st.write('Cumulative vertical distance traveled over time')
                        st.line_chart(distances[[time_unit_manual, 'y']], x=time_unit_manual)

                        st.write('Cumulative total distance traveled over time')
                        st.line_chart(distances[[time_unit_manual, 'total']], x=time_unit_manual)

                ### CREATE SUMMARY CSV HERE ###
                distances.to_csv("distance_" + video_name[:-4] + ".csv", index=False)
                z_manual.write("distance_" + video_name[:-4] + ".csv")
                z_manual.close()

                with open(f"{zip_name}.zip", "rb") as fp:
                    btn = st.download_button(label="Download results", data=fp, file_name=f"{zip_name}.zip",
                                             mime="application/zip")

# Second page, automatic
with (mode[1]):

    # Define all frontend components for Automatic tab
    st.title("Automatic Mode")
    st.markdown(
        "In automatic mode you can upload videos with their corresponding unlabeled DeepLabCut csv files and you will automatically get them labeled.")

    st.markdown("You can choose between the behaviours you want to consider: ")
    grooming_automatic = st.checkbox("Grooming", key='grooming_automatic')
    mid_rearing_automatic = st.checkbox("Mid rearing ", key='mid_rearing_automatic')
    wall_rearing_automatic = st.checkbox("Wall rearing ", key='wall_rearing_automatic')

    st.title("\n")

    uploaded_csvs = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    uploaded_videos = st.file_uploader("Upload Video files", type=["mp4"], accept_multiple_files=True)

    st.title("\n")

    with st.sidebar:
        time_unit = st.radio("Choose display unit", ("seconds", "frames"))

    # Process imputed data by the user
    video_names = set()  # Variable where we will save all the uploaded videos
    does_match = True  # Boolean variable to know if the uploaded data is correct

    # Check uploaded videos
    for uploaded_video in uploaded_videos:
        write_bytesio_to_file(uploaded_video.name, uploaded_video)  # Add video to video buffer in bytes
        video_names.add(uploaded_video.name[:-4] + ".mp4")  # Add video's names to a set of all names

    # Check uploaded behaviours
    # From the checkboxes, get the behaviours the user wants to predict
    behaviours = []
    if grooming_automatic:
        behaviours.append("grooming")
    if mid_rearing_automatic:
        behaviours.append("mid_rearing")
    if wall_rearing_automatic:
        behaviours.append("wall_rearing")
    # If there are not selected behaviours, return an error message
    if len(behaviours) == 0:
        st.write("Make sure you select at least one behaviour to analyze.")
        does_match = False

    # Check uploaded csvs
    if len(uploaded_csvs) > 0:
        tab_names = []  # List of csv files names
        for ind, uploaded_csv in enumerate(uploaded_csvs):
            tab_names.append(uploaded_csv.name[:-4])
        # Check if for each uploaded video there is a corresponding csv
        for csv_name in tab_names:
            corresponding_video_name = csv_name.split("_")[2] + ".mp4"
            if corresponding_video_name not in video_names or len(uploaded_csvs) != len(uploaded_videos):
                st.write("Make sure that each video has a corresponding .csv file and vice-versa.")
                does_match = False
    else:
        st.write("You haven't upload any videos yet!")
        does_match = False

    # If all videos have their corresponding csv, we can process them and generate the results
    if does_match:

        # Create the zip file where we will save the results
        z = zipfile.ZipFile(f"{zip_name}.zip", mode="w")

        for uploaded_csv in uploaded_csvs:
            write_bytesio_to_file(uploaded_csv.name, uploaded_csv)  # Add csv to video cav in bytes
        #    z.write(uploaded_csv.name)

        # For each uploaded video we create a tab for its results
        tabs = st.tabs(tab_names)

        # Get results for each video
        for index, tab in enumerate(tabs):
            with tab:
                # Get name of the video that will get analyzed
                video_name = uploaded_csvs[index].name.split("_")[2][:-4] + ".mp4"
                # Create data frame from video's csv
                df = pd.read_csv(uploaded_csvs[index])

                # Pass to the model the video and the csv and obtain prediction
                results = classify_video(df, video_name, "", behaviours)
                # Post-process results
                labels, distances, results = analyze_df(df, results)
                # Label video and get video rates
                fps = analyze_files(labels, video_name)

                distances['seconds'] = distances['frames'].map(lambda x: x / fps)

                st.write("Frames per second: ", fps)
                z.write("out_" + video_name)

                video_file = open("out_" + video_name, 'rb')

                st.video(video_file)

                st.write('Horizontal distance traveled over time')
                st.line_chart(distances[[time_unit, 'd_x']], x=time_unit)

                st.write('Vertical distance traveled over time')
                st.line_chart(distances[[time_unit, 'd_y']], x=time_unit)

                st.write('Total distance traveled over time')
                st.line_chart(distances[[time_unit, 'd_t']], x=time_unit)

                st.write('Cumulative horizontal distance traveled over time')
                st.line_chart(distances[[time_unit, 'cd_x']], x=time_unit)

                st.write('Cumulative vertical distance traveled over time')
                st.line_chart(distances[[time_unit, 'cd_y']], x=time_unit)

                st.write('Cumulative total distance traveled over time')
                st.line_chart(distances[[time_unit, 'cd_t']], x=time_unit)

        ### CREATE SUMMARY CSV HERE ###
        distances.to_csv("distance_" + video_name[:-4] + ".csv", index=False)
        results.to_csv("results" + video_name[:-4] + ".csv", index=False)

        z.write("distance_" + video_name[:-4] + ".csv")
        z.write("results" + video_name[:-4] + ".csv")

        z.close()

        with open(f"{zip_name}.zip", "rb") as fp:
            btn = st.download_button(label="Download results", key='download_manual', data=fp,
                                     file_name=f"{zip_name}.zip", mime="application/zip")

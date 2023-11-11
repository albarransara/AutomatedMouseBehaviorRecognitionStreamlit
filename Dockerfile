# pull official base image
#FROM python:3.12
#FROM tensorflow/tensorflow:2.1.0-py3
FROM tensorflow/tensorflow:latest

# set working directory
WORKDIR /app

RUN python3 -m pip install --upgrade pip

COPY ./src/requirements.txt /app/requirements.txt

# Install requirement 
RUN pip install -r /app/requirements.txt

COPY src/ /app/src/

EXPOSE 8501

CMD ["python3", "-m", "streamlit", "run", "./src/frontend.py"]
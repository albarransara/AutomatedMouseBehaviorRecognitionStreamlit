FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# set working directory
WORKDIR /app

# Install additional packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    wget \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install requirements
COPY ./src/requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Copy source code
COPY ./src/ /app/src/

# Open streamlit port
EXPOSE 8501

# Run Streamlit project
CMD ["python3", "-m", "streamlit", "run", "./src/frontend.py", "--server.port=8501"]
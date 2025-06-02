FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "ds_agent", "/bin/bash", "-c"]

# Copy project files (including protobuf directory)
COPY . .

# Generate protobuf classes
RUN conda run -n ds_agent python3 -m grpc_tools.protoc -I protobuf \
    --python_out=utils/salute_speech \
    --grpc_python_out=utils/salute_speech \
    protobuf/recognition.proto \
    protobuf/storage.proto \
    protobuf/task.proto \
    protobuf/synthesis.proto

# Set up the container's entry command
EXPOSE 8501
ENTRYPOINT ["conda", "run", "-n", "ds_agent", "streamlit", "run", "app.py", "--server.address=0.0.0.0"] 
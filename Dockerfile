# Use an official Python 3.8 image
FROM python:3.8

# Set the working directory in the docker container
WORKDIR /app

# We copy just the pyproject.toml and poetry.lock files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install necessary dependencies using poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false \
&& poetry install --no-interaction --no-ansi --no-dev

# Copy the remaining application files into the container at /app
# This includes our main Streamlit application script and any other necessary files
COPY . /app

# Expose the port Streamlit runs on, which is 8501 by default
# This makes the port available to other services and users
EXPOSE 8501

# Define the command to run our Streamlit app
# This CMD instruction is executed when the Docker container starts up
CMD ["streamlit", "run", "streamlit_app.py"]

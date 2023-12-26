# Use an official Python 3.8 image
FROM python:3.8

# Set the working directory in the Docker container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install dependencies using poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false \
&& poetry install --no-interaction --no-ansi --no-dev

# Copy the remaining application files into the container at /app
# This includes our main Streamlit application script and any other necessary files
COPY . /app

# Set the environment variable for the port
# Cloud Run will set this environment variable automatically,
# but the default is also set here for local testing purposes
ENV PORT=8501

# Expose the port the app runs on
EXPOSE $PORT

# Define the command to run the Streamlit app
CMD ["sh", "-c", "streamlit run --server.port $PORT streamlit_app.py"]

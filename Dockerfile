# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1-python3.10

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java and required utilities
RUN rm -f /etc/apt/sources.list.d/pgdg.list && \
    sed -i \
      's|^deb http://deb.debian.org/debian-security bullseye-security main$|deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20260903T220410Z bullseye-security main|' \
      /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
        bash && \
    rm -rf /var/lib/apt/lists/*

# Configure Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app

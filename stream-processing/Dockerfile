# ============================================================
# Flink Stream Processing - PyFlink Docker Image
# ============================================================
FROM flink:1.18-java11

# Python install
USER root

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python symbolic link
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Working directory
WORKDIR /app

# Python packages
COPY requirements.txt /app/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Source code
COPY src/ /app/src/

# PyFlink environment
ENV PYFLINK_PYTHON=/usr/bin/python3
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Flink config directory
RUN mkdir -p /opt/flink/usrlib

# Entrypoint
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

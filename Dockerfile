FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash agent

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code
COPY src/ /app/src/

# Run as non-root
USER agent
WORKDIR /app

# Agent state is mounted, not baked in
VOLUME ["/home/agent/.agent"]

CMD ["python", "-m", "src.main"]

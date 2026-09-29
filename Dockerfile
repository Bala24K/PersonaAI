FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies & C++ build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Compile C++17 Text Analyzer component
RUN g++ -std=c++17 -Icpp/include cpp/src/analyzer.cpp cpp/src/main.cpp -o cpp/persona_cpp_analyzer && \
    chmod +x cpp/persona_cpp_analyzer

# Create non-root runtime user for production security
RUN useradd -m -u 1000 persona && \
    chown -R persona:persona /app

USER persona

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

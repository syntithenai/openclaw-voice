FROM rocm/pytorch:latest

WORKDIR /app

RUN getent group video >/dev/null || groupadd -r video
RUN getent group render >/dev/null || groupadd -r render

# Set ROCm environment variables for HIP initialization
ENV HIP_VISIBLE_DEVICES=0
ENV ROCM_HOME=/opt/rocm
ENV PATH=${ROCM_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_HOME}/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

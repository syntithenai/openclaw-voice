FROM rocm/pytorch:latest

WORKDIR /app

RUN getent group video >/dev/null || groupadd -r video
RUN getent group render >/dev/null || groupadd -r render

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn onnxruntime-rocm \
	&& pip install --no-cache-dir --no-deps piper-tts

COPY app.py ./app.py

# Create directory for piper models
RUN mkdir -p /root/.local/share/piper

EXPOSE 10001
ENTRYPOINT ["python", "-m"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10001"]

FROM python:3.12-slim-bullseye

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
	&& apt-get update && apt-get upgrade -y --no-install-recommends \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app

RUN pip install -e .

EXPOSE 8501
CMD ["python", "marker_app.py"]
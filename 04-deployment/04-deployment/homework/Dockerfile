FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]


ENTRYPOINT [ "python", "starter.py" ]

# docker build -t durationpredictor:v1 --platform=linux/arm64 .
# docker run -it -v ./output/yellow:/app/output/yellow durationpredictor:v1 yellow 2022 5 -- Runs for April 2022 (relativedelta(months=1) applied)
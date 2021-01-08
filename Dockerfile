FROM python:3.7-slim

WORKDIR /app
COPY . /app/

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

CMD ["python3", "main_3.py"]

FROM python:3.6-slim

ENV HOME=/app

WORKDIR $HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    git

COPY requirements.txt $HOME/

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . $HOME/
RUN pip install -e .

CMD ["pytest", "--color=yes", "-s", "tests/test.py"]

FROM python:3.13

WORKDIR /test

COPY ./requirements.txt /test/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /test/requirements.txt

CMD ["uvicorn", "fa_test:app", "--host", "0.0.0.0", "--port", "8000"]

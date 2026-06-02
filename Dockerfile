FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi uvicorn pandas numpy xgboost pypickle boto3 --no-cache-dir

COPY . .

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_BUCKET_NAME

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_BUCKET_NAME=${AWS_BUCKET_NAME}

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
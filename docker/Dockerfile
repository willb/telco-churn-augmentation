FROM mattf/spark:alpha-3.0.2-hadoop3.2-accelerator0.4.1

RUN apt-get update && apt-get install -y git pipenv

WORKDIR /workspace

RUN git clone -b v0.7 https://github.com/willb/telco-churn-augmentation .

RUN sed -i -e '/altair/d' -e '/pyspark/d' Pipfile

RUN pipenv install

COPY start.sh /workspace

RUN chmod 755 /workspace/start.sh

CMD ["./start.sh"]


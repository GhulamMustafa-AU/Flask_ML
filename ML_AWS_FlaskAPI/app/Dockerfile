FROM python:3-onbuild

WORKDIR /var/app
COPY . .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install --upgrade pip

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
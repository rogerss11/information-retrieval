FROM python:3.11-slim
WORKDIR /app
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
COPY courses.json /app/courses.json
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
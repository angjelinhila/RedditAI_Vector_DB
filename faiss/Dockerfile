FROM python:3.9

WORKDIR /app
RUN pip install faiss-cpu numpy flask

COPY faiss_service.py .

CMD ["python", "faiss_service.py"]

FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash", "-lc", "streamlit run app/app.py -- --checkpoint_path checkpoints/best.ckpt"]

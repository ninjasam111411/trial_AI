# Runs Neural as a website (works on Hugging Face Spaces, Render, Railway, etc.)
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY --chown=user . .
USER user

# Train a starter model while building, so the site is ready right away.
RUN python build_dataset.py && \
    python train.py --steps 1500 --context-size 128 --embedding-size 128 --heads 4 --layers 3

ENV HOST=0.0.0.0 PORT=7860 NO_BROWSER=1
EXPOSE 7860
# One worker so the whole app shares one brain; threads handle many users.
CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "120", "-b", "0.0.0.0:7860", "app:app"]

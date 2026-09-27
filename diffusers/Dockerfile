# Forkline の汎用 Diffusers ワーカー。モデルの重みはイメージに入れず、起動時に Hugging Face から読む
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/runpod-volume/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 画像・動画の前処理 (Qwen3-VL の processor など) に torchvision が要る。ベースのイメージに含まれるものを使い、
# pip で入れ直すと torch と版がずれるため、無ければビルドを止める。量子化の torchao が入っていることと、
# 依存を入れても torch の版 (2.9.1) が変わっていないことも確かめる
RUN python -c "import torch, torchvision, torchao; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'torchao', torchao.__version__); assert torch.__version__.startswith('2.9.1'); assert torchao.__version__ >= '0.16'"
COPY handler.py .

CMD ["python", "-u", "handler.py"]

"""
ワーカーの読み込みと失敗の扱いを、GPU なしで偽のパイプラインを使って確かめる。
実行: python diffusers/test_engine.py (requirements.txt の依存と CPU 版の torch が要る)
- 重みは GPU へ直接読み込み、オフロードは CPU_OFFLOAD=1 のときだけ
- メモリが足りなければ、読み直さずに失敗の要約を返す (読み込みの失敗でもワーカーは終了しない)
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

HANDLER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handler.py")

# ---- GPU と外部サービスの代わり
# torch.Generator を差し替える前に、ワーカーが使う diffusers の部分を読み込んでおく (遅延読み込みのため)
from diffusers import ComponentsManager, DiffusionPipeline, ModularPipeline  # noqa: E402,F401
from diffusers.utils import encode_video  # noqa: E402,F401

# GPU がある環境として動かす (読み込み先は cuda、生成は CPU の偽物)
torch.cuda.is_available = lambda: True
torch.cuda.is_bf16_supported = lambda: True
torch.cuda.get_device_name = lambda _=0: "Test GPU"
torch.cuda.get_device_properties = lambda _=0: SimpleNamespace(total_memory=80e9)
torch.cuda.memory_allocated = lambda: 70e9
torch.cuda.max_memory_allocated = lambda: 79e9
# Windows では開いたままの一時ファイルを別の処理から開けないため、テストでは閉じてから使う (Linux のワーカーでは不要)
import functools  # noqa: E402
import tempfile  # noqa: E402

tempfile.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete=False)
real_generator = torch.Generator
torch.Generator = lambda device=None: real_generator()
sys.modules["runpod"] = types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda _: None))

events = []


class FakePipeline:
    fail_load_on_gpu = False
    fail_generate_on_gpu = False
    fail_generate_offloaded = False

    def __init__(self, device_map):
        self.offloaded = False
        self.device_map = device_map
        self.components = {"unet": torch.nn.Linear(2, 2)}

    @classmethod
    def from_pretrained(cls, repo, dtype=None, token=None, device_map=None):
        events.append(f"load device_map={device_map}")
        if device_map == "cuda" and cls.fail_load_on_gpu:
            raise torch.OutOfMemoryError("load")
        return cls(device_map)

    def enable_model_cpu_offload(self):
        self.offloaded = True
        events.append("offload")

    def __call__(self, prompt=None, generator=None, num_images_per_prompt=1, output_type=None, num_frames=None):
        seed = generator.initial_seed()
        on_gpu = not self.offloaded
        if (on_gpu and FakePipeline.fail_generate_on_gpu) or (not on_gpu and FakePipeline.fail_generate_offloaded):
            events.append(f"generate OOM seed={seed}")
            raise torch.OutOfMemoryError("generate")
        events.append(f"generate ok seed={seed} offloaded={self.offloaded}")
        if output_type == "np":
            return SimpleNamespace(frames=[np.zeros((9, 32, 32, 3), dtype=np.float32)])
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))] * num_images_per_prompt)


def load_handler(task, offload=False, **flags):
    for key, value in flags.items():
        setattr(FakePipeline, key, value)
    events.clear()
    os.environ["MODEL_NAME"] = "test/model"
    os.environ["TASK"] = task
    os.environ["CPU_OFFLOAD"] = "1" if offload else ""
    import diffusers
    import huggingface_hub

    diffusers.DiffusionPipeline = FakePipeline
    huggingface_hub.model_info = lambda *a, **k: SimpleNamespace(
        pipeline_tag=task, siblings=[SimpleNamespace(rfilename="model_index.json")]
    )
    spec = importlib.util.spec_from_file_location("handler", HANDLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(name, condition):
    print(("ok  " if condition else "FAIL"), name)
    if not condition:
        print("    events:", events)
        sys.exit(1)


# 1. 生成中にメモリ不足 → 読み直さずに、失敗の要約を返す
h = load_handler("text-to-image", fail_load_on_gpu=False, fail_generate_on_gpu=True, fail_generate_offloaded=False)
result = h.handler({"input": {"prompt": "fox", "seed": 42}})
check("generation OOM does not reload", events == ["load device_map=cuda", "generate OOM seed=42"])
check("generation OOM returns a summary", result["error"].startswith("OutOfMemoryError: generate | while generating at "))
check("summary names the loading state", "StandardPipelineAdapter on GPU" in result["error"])
check("summary includes GPU memory", "Test GPU used 70.0/80.0GB peak 79.0GB" in result["error"])
check("summary fits in the 300 characters Forkline keeps", len(result["error"]) <= 300)
print("    e.g.", result["error"])

# 2. 読み込みでメモリ不足 → 終了せず、各ジョブに読み込みの失敗を返す
h = load_handler("text-to-image", fail_load_on_gpu=True, fail_generate_on_gpu=False, fail_generate_offloaded=False)
check("load OOM does not fall back to offload", events == ["load device_map=cuda"])
result = h.handler({"input": {"prompt": "fox", "seed": 1}})
check("jobs get the load failure", result["error"].startswith("OutOfMemoryError: load | while loading at "))

# 3. CPU_OFFLOAD=1 のときだけ、動かす部品だけを GPU に載せる
h = load_handler("text-to-image", offload=True, fail_load_on_gpu=False, fail_generate_on_gpu=False, fail_generate_offloaded=False)
result = h.handler({"input": {"prompt": "fox", "seed": 5, "num_images": 2}})
check("offload is used only when asked", events == ["load device_map=None", "offload", "generate ok seed=5 offloaded=True"])
check("returns images", len(result["images"]) == 2 and result["seed"] == 5)

# 4. オフロード中に足りなくても、要約を返す
h = load_handler("text-to-image", offload=True, fail_load_on_gpu=False, fail_generate_on_gpu=False, fail_generate_offloaded=True)
result = h.handler({"input": {"prompt": "fox", "seed": 1}})
check("offloaded OOM names the offload state", "StandardPipelineAdapter offload" in result["error"])

# 5. 動画 (音声なし) は MP4 にする
h = load_handler("text-to-video", fail_load_on_gpu=False, fail_generate_on_gpu=False, fail_generate_offloaded=False)
result = h.handler({"input": {"prompt": "fox", "seed": 3, "num_frames": 9, "fps": 8}})
check("video is an mp4 data URI", result["video"].startswith("data:video/mp4;base64,") and result["fps"] == 8 and result["audio"] is False)

# 6. 必須の入力が無ければエラーを返す
check("missing prompt is rejected", "error" in h.handler({"input": {}}))
print("all passed")

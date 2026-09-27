"""
Forkline の汎用 Diffusers ワーカー (RunPod Serverless)。

環境変数 MODEL_NAME の Hugging Face リポジトリを Diffusers で読み込み、
テキストや画像から画像・動画を作る。モデルごとにイメージを作らなくてよいように、
種類 (text-to-image / text-to-video / image-to-video) はリポジトリの情報から決める。

入力:  {"prompt": "...", ...}  (項目の定義とフォームは Forkline が持つ。ここでは必須の項目だけを確かめる)
出力:  画像 {"images": ["data:image/png;base64,..."], "seed": 1}
       動画 {"video": "data:video/mp4;base64,...", "fps": 16, "seed": 1}
"""

import base64
import inspect
import io
import os
import random
import tempfile
import traceback
import urllib.request
from dataclasses import dataclass

import runpod
import torch
from diffusers import ComponentsManager, DiffusionPipeline, ModularPipeline
from diffusers.utils import encode_video
from PIL import Image
from huggingface_hub import model_info

MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ.get("HF_TOKEN") or None
MAX_IMAGE_BYTES = 20 * 1024 * 1024

PIPELINE_ARGS = ["prompt", "negative_prompt", "width", "height", "num_inference_steps", "guidance_scale", "num_frames"]
# 種類ごとの必須の入力。モジュラー形式では、この入力で動くワークフローを選ぶ
TASK_INPUTS = {
    "text-to-image": {"prompt"},
    "text-to-video": {"prompt"},
    "image-to-video": {"prompt", "image"},
}

# ---- リポジトリと実行環境 (起動時に1度だけ決める)

REPO = model_info(MODEL_NAME, token=HF_TOKEN)
# 種類は Forkline の設定 (環境変数 TASK) に従い、無ければ Hugging Face の pipeline_tag
TASK = os.environ.get("TASK") or REPO.pipeline_tag
if TASK not in TASK_INPUTS:
    raise RuntimeError(f"Unsupported task for {MODEL_NAME}: {TASK}")
# modular_model_index.json があればモジュラー形式として読む (従来の model_index.json が並んでいても優先する)
MODULAR = any(file.rfilename == "modular_model_index.json" for file in REPO.siblings or [])
# 本番は GPU。GPU の無い環境では CPU で動かす (小さな検証用のモデルで、実際のパイプラインを通して確かめるため)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32


@dataclass
class Video:
    frames: object
    audio: object | None = None
    sample_rate: int | None = None


class StandardPipelineAdapter:
    """従来の形式 (model_index.json)。受け付ける入力は __call__ の引数から分かる"""

    def __init__(self, offload: bool):
        load_args = {"dtype": DTYPE, "token": HF_TOKEN}
        if offload:
            self.pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, **load_args)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, **load_args, device_map=DEVICE)
        self.inputs = set(inspect.signature(self.pipe.__call__).parameters)
        self.components = self.pipe.components
        self.fixed_fps = None

    def images(self, **kwargs) -> list:
        return self.pipe(**kwargs).images

    def video(self, **kwargs) -> Video:
        result = self.pipe(**kwargs, output_type="np")
        audio = getattr(result, "audio", None)
        # 音声も作るモデル (LTX-2 など) は、音声の変換器 (vocoder) がサンプリングレートを持つ
        vocoder = getattr(self.pipe, "vocoder", None)
        if audio is None or vocoder is None:
            return Video(result.frames[0])
        return Video(result.frames[0], audio[0], vocoder.config.output_sampling_rate)


class ModularPipelineAdapter:
    """
    モジュラー形式。1つのリポジトリに複数のワークフロー (MiniMax-H3 なら t2va / fl2va / ref2va) があり、
    種類の入力で動くワークフローの部品だけを読む。受け付ける入力と出力はブロックが宣言している
    """

    def __init__(self, offload: bool):
        manager = ComponentsManager() if offload else None
        self.pipe = ModularPipeline.from_pretrained(MODEL_NAME, token=HF_TOKEN, components_manager=manager)
        blocks = self.pipe.blocks
        load_args = {"dtype": DTYPE, "token": HF_TOKEN}
        # 重みを持つ部品 (モデル)。GPU へ直接読み込むのはこれだけで、トークナイザーなどには渡さない
        models = [
            spec
            for spec in blocks.expected_components
            if isinstance(spec.type_hint, type) and issubclass(spec.type_hint, torch.nn.Module)
        ]
        if not offload:
            load_args["device_map"] = {spec.name: DEVICE for spec in models}
        workflow = pick_workflow(blocks, TASK_INPUTS[TASK])
        self.pipe.load_components(workflow=workflow, **load_args)
        # 読み込めなかった部品は、エラーをログに出すだけで空のまま残る (生成の途中で分かりにくいエラーになる)。
        # ワークフローが使う部品がそろっているかを、ここで確かめる
        needed = blocks.get_workflow(workflow).expected_components if workflow else blocks.expected_components
        missing = [spec.name for spec in needed if self.pipe.components.get(spec.name) is None]
        if missing:
            raise RuntimeError(f"Failed to load components of {MODEL_NAME}: {', '.join(missing)}")
        if manager is not None:
            # 部品をすべて読んでから有効にする。ComponentsManager は部品が加わるたびにオフロードを既定の戦略で
            # 有効にし直すため、読む前に指定した戦略は上書きされる (LTX-2.5 はこれで GPU から部品が外れなかった)
            manager.enable_auto_cpu_offload(device=DEVICE, offload_strategy=offload_all_others)
        self.inputs = set(blocks.input_names)
        self.outputs = set(blocks.output_names)
        self.components = self.pipe.components
        # フレームレートが固定のモデル (MiniMax-H3 は 24fps) はパイプラインが持つ
        self.fixed_fps = getattr(self.pipe, "fps", None)

    def images(self, **kwargs) -> list:
        return self.pipe(**kwargs, output="images")

    def video(self, **kwargs) -> Video:
        wanted = [name for name in ("videos", "audio", "sampling_rate") if name in self.outputs]
        result = self.pipe(**kwargs, output_type="np", output=wanted)
        audio = result.get("audio")
        if audio is None or not result.get("sampling_rate"):
            return Video(result["videos"][0])
        return Video(result["videos"][0], audio[0], result["sampling_rate"])


def offload_all_others(hooks, **_):
    """
    部品を動かす前に、GPU に載っているほかの部品をすべて外す (従来の形式の enable_model_cpu_offload と同じ振る舞い)。
    既定の戦略は数 GB の空きしか残さないため、動画の生成中の計算でメモリが足りなくなる
    """
    return hooks


def pick_workflow(blocks, provided: set):
    """
    渡す入力だけで動くワークフローのうち、最も多くの入力を使うものを選ぶ (画像を渡すなら画像から作るもの)。
    ワークフローを選ぶ条件はブロックの _workflow_map が宣言している (公開の API は名前の一覧だけ)
    """
    triggers = getattr(blocks, "_workflow_map", None)
    if not triggers:
        return None
    best, best_size = None, -1
    for name, conditions in triggers.items():
        for condition in conditions if isinstance(conditions, tuple) else (conditions,):
            required = {key for key, needed in condition.items() if needed}
            if required <= provided and len(required) > best_size:
                best, best_size = name, len(required)
    if best is None:
        raise RuntimeError(f"No workflow of {MODEL_NAME} runs with {sorted(provided)}")
    return best


PIPELINE_CLASS = ModularPipelineAdapter if MODULAR else StandardPipelineAdapter
# 動かす部品だけを GPU に載せる形 (オフロード)。遅くなるので、モデルの設定で選んだときだけ使う
OFFLOAD = os.environ.get("CPU_OFFLOAD") == "1"


def load_pipeline():
    """重みは GPU へ直接読み込む (CPU 側に全部を読み込んでから移すと、大きいモデルでホストのメモリが尽きる)"""
    pipe = PIPELINE_CLASS(offload=OFFLOAD)
    weights = sum(
        sum(p.numel() * p.element_size() for p in component.parameters())
        for component in pipe.components.values()
        if isinstance(component, torch.nn.Module)
    )
    print(
        f"loaded {MODEL_NAME} ({PIPELINE_CLASS.__name__}) task={TASK} "
        f"weights={weights / 1e9:.1f}GB offload={OFFLOAD}"
    )
    return pipe


def describe_failure(error: Exception, stage: str) -> str:
    """
    失敗の理由を、ワーカーが終わるとログが消えても分かるように短くまとめる (Forkline はジョブのエラーとして先頭の
    300 文字を残す)。原因、起きた段階と場所、読み込みの形、GPU のメモリの順に並べる
    """
    message = str(error).strip().splitlines()[0] if str(error).strip() else ""
    frames = traceback.extract_tb(error.__traceback__)
    where = f"{os.path.basename(frames[-1].filename)}:{frames[-1].name}" if frames else "?"
    parts = [
        f"{type(error).__name__}: {message[:120]}",
        f"while {stage} at {where}",
        f"{PIPELINE_CLASS.__name__} {'offload' if OFFLOAD else 'on GPU'}",
    ]
    if DEVICE == "cuda":
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            parts.append(
                f"{torch.cuda.get_device_name(0)} used {torch.cuda.memory_allocated() / 1e9:.1f}"
                f"/{total / 1e9:.1f}GB peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            )
        except Exception:  # 要約を作れなくても、元のエラーは返す
            pass
    return " | ".join(parts)


# ---- 入出力


def load_input_image(source: str):
    """画像は URL か data URI で受け取る。"""
    if source.startswith("data:"):
        data = base64.b64decode(source.split(",", 1)[1])
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image is too large")
        return _image_from_bytes(data)
    if not source.startswith(("https://", "http://")):
        raise ValueError("image must be a URL or a data URI")
    with urllib.request.urlopen(source, timeout=30) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image is too large")
    return _image_from_bytes(data)


def _image_from_bytes(data: bytes):
    return Image.open(io.BytesIO(data)).convert("RGB")


def to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def generate(pipe, params: dict, seed: int):
    # 指定された項目のうち、このパイプラインが受け付けるものだけを渡し、ほかは既定値に任せる
    kwargs = {
        name: params[name]
        for name in PIPELINE_ARGS
        if params.get(name) is not None and name in pipe.inputs
    }
    kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)
    if TASK == "image-to-video":
        kwargs["image"] = load_input_image(params["image"])

    if TASK == "text-to-image":
        if "num_images_per_prompt" in pipe.inputs:
            kwargs["num_images_per_prompt"] = int(params.get("num_images") or 1)
        encoded = []
        for image in pipe.images(**kwargs):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(to_data_uri(buffer.getvalue(), "image/png"))
        return {"images": encoded, "seed": seed}

    # フレームレートが固定のモデルはその値で書き出す。生成に使うモデル (LTX-2 は frame_rate) には指定の値を渡す
    fps = int(pipe.fixed_fps or params.get("fps") or 16)
    if "frame_rate" in pipe.inputs:
        kwargs["frame_rate"] = float(fps)
    video = pipe.video(**kwargs)
    # 音声も作るモデル (LTX-2、MiniMax-H3 など) は、音声つきの MP4 にする
    audio_args = {}
    if video.audio is not None:
        audio_args = {"audio": video.audio.float().cpu(), "audio_sample_rate": int(video.sample_rate)}
    with tempfile.NamedTemporaryFile(suffix=".mp4") as file:
        encode_video(video.frames, fps=fps, output_path=file.name, **audio_args)
        data = open(file.name, "rb").read()
    return {"video": to_data_uri(data, "video/mp4"), "fps": fps, "seed": seed, "audio": bool(audio_args)}


# 読み込みに失敗しても終了しない (終了すると RunPod が起動を繰り返し、料金だけがかかる)。
# 代わりに各ジョブへ失敗の要約を返し、モデルの作者が原因 (GPU のメモリ不足など) を知れるようにする
PIPE, LOAD_FAILURE = None, None
try:
    PIPE = load_pipeline()
except Exception as error:
    traceback.print_exc()
    LOAD_FAILURE = describe_failure(error, "loading")


def handler(job):
    params = job.get("input") or {}
    missing = [name for name in sorted(TASK_INPUTS[TASK]) if not params.get(name)]
    if missing:
        return {"error": f"missing required input: {', '.join(missing)}"}

    # エラーを返すと、RunPod はジョブを失敗として扱い、この文をエラーとして返す
    if LOAD_FAILURE:
        return {"error": LOAD_FAILURE}
    seed = params.get("seed")
    seed = random.randint(0, 2**32 - 1) if seed is None else int(seed)
    try:
        return generate(PIPE, params, seed)
    except Exception as error:
        traceback.print_exc()
        return {"error": describe_failure(error, "generating")}


runpod.serverless.start({"handler": handler})

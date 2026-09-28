# forkline-workers

Forkline が GPU (RunPod Serverless) で使う汎用ワーカー。

## diffusers

Hugging Face の Diffusers 形式のモデル (text-to-image / text-to-video / image-to-video) を、
環境変数で指定したリポジトリから読み込んで動かす。モデルごとにイメージを作る必要はない。

- **形式**: `modular_model_index.json` があればモジュラー形式 (MiniMax-H3、LTX-2.5 など)、無ければ従来の形式
  (`model_index.json`) として読む。モジュラー形式は、種類の入力 (文章だけ / 画像つき) で動くワークフローを選び、
  その部品だけを読み込む。
  text-to-video は、モデルが受け付ければ最初のフレームの画像 (`image`) も任意で受け取る (MiniMax-H3 など)。
- **GPU のメモリ**: 重みは GPU へ直接読み込む。`CPU_OFFLOAD=1` のときだけ、動かす部品だけを GPU に載せる
  (遅くなるが、GPU に収まらないモデルも動く)。メモリが足りなければ自動では切り替えず、失敗の要約を返す。
- **失敗**: 原因、起きた段階と場所、読み込みの形、GPU のメモリを短くまとめてジョブのエラーとして返す。
  読み込みに失敗してもワーカーは終了せず、各ジョブにその要約を返す (終了すると起動を繰り返して料金だけがかかるため)。
- **確認**: `python diffusers/test_engine.py` で、読み込みと失敗の扱いを GPU なしで確かめられる。

| 環境変数 | 内容 |
|---|---|
| `MODEL_NAME` | Hugging Face のリポジトリ (例: `stabilityai/sdxl-turbo`) |
| `HF_TOKEN` | 非公開・利用申請が必要なリポジトリ用 (任意) |
| `TASK` | 種類 (任意。Forkline は設定の種類を渡す。未指定なら Hugging Face の `pipeline_tag`) |
| `CPU_OFFLOAD` | `1` で動かす部品だけを GPU に載せる。GPU のメモリに収まらないモデル用 (遅くなる) |

入力は `{"prompt": "..."}` など。項目の定義 (JSON Schema) とデモのフォームは Forkline が持ち、ワーカーは必須の項目だけを確かめる。
出力の画像・動画は data URI で返し、Forkline が保存して URL に置き換える。

イメージ: `ghcr.io/user-s0ma/forkline-diffusers:latest` (main に push すると GitHub Actions でビルド)

### 量子化済みの重み

GPU に収まらない大きなモデル (MiniMax-H3 など) は、量子化済みの Diffusers 形式のリポジトリを `MODEL_NAME` に
指定する。量子化の設定は各部品の `config.json` に入っているので、ワーカーは何も指定せずに読む。torchao で
量子化した重みの読み込みには torchao 0.16.0 以上が要る (イメージに入っている)。


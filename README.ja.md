# lerobot-wandb

[English](./README.md)

`lerobot-wandb` は、LeRobot で作ったデータセット、学習済みポリシー、ロールアウト結果を W&B Artifacts で受け渡すための補助 CLI です。upstream [LeRobot](https://github.com/huggingface/lerobot) 自体には手を加えません。

![LeRobot と W&B のワークフロー](./assets/wandb-workflow-overview-ja.jpg)

この図は LeRobot と W&B を組み合わせた全体の流れを示しています。`lerobot-wandb` が担うのは、記録・学習・ロールアウトの前後にある Artifact 操作です。ロボットを動かすコマンドには通常の LeRobot を使います。

## 役割

`lerobot-wandb` は、既存の LeRobot 環境に追加して使います。LeRobot の処理を包み込むのではなく、ローカルディレクトリと W&B Artifact の間をつなぎます。

主な機能は次のとおりです。

- データセット、モデル、ロールアウトデータセットを W&B Artifacts としてアップロード／ダウンロードする
- アップロード前とダウンロード後にローカルディレクトリを検証する
- canonical な動画を変更せず、ブラウザ確認用のプレビューを作成する
- 指定した Artifact 参照と、実際に解決された immutable version の両方を記録する
- ロールアウト結果を、評価に使ったモデルバージョンへ紐付ける
- 評価済みモデルを alias または W&B Registry で昇格する

パッケージの所有範囲は明確に分けています。`lerobot-wandb` は LeRobot 本体とは別の Python distribution であり、LeRobot のネイティブプラグインではありません。

- distribution: `lerobot-wandb`
- Python package: `lerobot_wandb`
- CLI command: `lerobot-wandb`

ロボット制御、記録、学習、ポリシー実行は引き続き LeRobot が担当します。

| 作業 | コマンド | 担当 |
| --- | --- | --- |
| 教示データの記録 | `lerobot-record` | LeRobot |
| データセットの upload / download | `lerobot-wandb dataset ...` | `lerobot-wandb` |
| ポリシー学習 | `lerobot-train` | LeRobot |
| モデルの upload / download / promote | `lerobot-wandb model ...` | `lerobot-wandb` |
| ロボット上でのポリシー実行 | `lerobot-rollout` | LeRobot |
| ロールアウト結果の公開 | `lerobot-wandb rollout upload` | `lerobot-wandb` |

両者の受け渡しはローカルディレクトリを介して行います。W&B が入るのは完成した Artifact の保存・取得部分だけで、ロボットの制御ループには入りません。

## 動作条件

- Python 3.12 以降
- W&B アカウント、および Artifact の送受信時に利用できるネットワーク
- LeRobot のデータセットや動画を扱うコマンドでは upstream LeRobot `>=0.6.1,<0.6.2`
- 利用するロボット、カメラ、動画処理に必要な LeRobot 側の依存関係

現在のパッケージバージョンは `0.1.0` です。まだ PyPI には公開していないため、GitHub からインストールします。

## インストール

`lerobot-wandb` は LeRobot と同じ Python 環境に入れてください。LeRobot のデータを扱うコマンドは、その環境にある LeRobot を import します。

すでに LeRobot の checkout と `.venv` がある場合:

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
wandb login
lerobot-wandb --help
```

新しく環境を作る場合は、1 つの `.venv` に LeRobot と `lerobot-wandb` を入れます。

```bash
mkdir lerobot-workspace
cd lerobot-workspace

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

uv pip install "lerobot[core_scripts,training,feetech]==0.6.1"
uv pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"

wandb login
lerobot-info
lerobot-wandb --help
```

ここで `feetech` を入れているのは、後の例で SO-101 を使うためです。実際には使用するハードウェアに合う LeRobot extra を選んでください。FFmpeg、PyTorch など OS 依存のセットアップは upstream の [LeRobot インストールガイド](https://huggingface.co/docs/lerobot/installation) に従います。

base package から LeRobot を hard dependency にしていないのは、依存解決によって既存の LeRobot が意図せず入れ替わるのを避けるためです。LeRobot を必要とするコマンドは実行時にバージョンを確認します。`--allow-unsupported-lerobot` はバージョンチェックを回避するための実験用オプションであり、互換性を保証するものではありません。

## アンインストール

LeRobot と同じ環境を有効化し、companion package だけを削除します。

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip uninstall lerobot-wandb
```

削除対象は `lerobot-wandb` distribution、`lerobot_wandb` package、`lerobot-wandb` CLI command です。LeRobot はインストールされたままで、変更されません。`wandb`、`datasets`、`pandas` など共有している依存パッケージも自動では削除しません。

アンインストールしても、ローカルの dataset、download / materialize 済み Artifact、model、rollout directory、training output、sidecar metadata などのユーザーデータは残ります。W&B 上の Artifact、Run、Registry object は削除しません。W&B の認証情報や設定も削除しません。

## ワークフロー例の設定

以下のコマンドでは同じ名前とパスを繰り返し使うので、最初に環境変数へまとめておきます。

```bash
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="your-wandb-project"
export DATASET_NAME="your-dataset"
export POLICY_NAME="your-policy"
export ROLLOUT_NAME="your-rollout"

export DATASET_ROOT="./data/$DATASET_NAME"
export TRAIN_DATASET_ROOT="./datasets/$DATASET_NAME"
export TRAIN_OUTPUT="./outputs/train/$POLICY_NAME"
export POLICY_ROOT="$TRAIN_OUTPUT/checkpoints/last/pretrained_model"
export DOWNLOADED_POLICY_ROOT="./policies/$POLICY_NAME-candidate"
export ROLLOUT_ROOT="./data/$ROLLOUT_NAME"
```

これらは説明用の名前であり、`lerobot-wandb` が強制する命名規則ではありません。

コマンド例は Linux と Bash を前提にしています。Windows では同じ LeRobot 環境を PowerShell 用の方法で有効化し、`/dev/ttyACM*` を実際の `COM` port に置き換えてください。

## 最短手順: W&B のデータセットから学習する

すでに W&B にデータセットがあるなら、download → train → model upload の 3 段階です。

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$DATASET_NAME:raw" \
  --root "$TRAIN_DATASET_ROOT"

lerobot-train \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$TRAIN_DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="$TRAIN_OUTPUT" \
  --steps=100000 \
  --policy.push_to_hub=false

lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

checkpoint の実際の配置はポリシーと学習設定で変わります。upload 前に LeRobot の出力先を確認してください。

## 一連の流れ

Artifact 側の手順はロボット機種に依存しません。以下の robot-facing command だけ、具体例として SO-101 を使います。

### 1. LeRobot で教示データを記録する

記録は通常の LeRobot だけで完結します。

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.num_episodes=30 \
  --dataset.push_to_hub=false
```

`repo_id` は LeRobot 側のローカル識別子です。`root` には、後で `lerobot-wandb` が検証して upload するディレクトリを指定します。

### 2. データセットを W&B に公開する

```bash
lerobot-wandb dataset upload \
  --root "$DATASET_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$DATASET_NAME" \
  --alias raw
```

W&B Run を開始する前にデータセットを検証します。upload が終わると、次のような immutable reference が表示されます。

```text
your-wandb-entity/your-wandb-project/your-dataset:v0
```

再現性が必要な記録には alias ではなく `vN` を残してください。`raw` のような alias は後から別バージョンを指すことがあります。

#### データセット動画のプレビュー

canonical な動画は変更せず、そのまま Artifact に保存します。ブラウザ確認用のプレビューは別ファイルとして作成します。

対象の選び方:

- default: 各 camera から決定的に選んだ代表 episode を 1 件
- 指定 episode: `--preview-episode INDEX` を必要な回数だけ指定
- 全 episode / 全 camera: `--preview-all`
- preview なし: `--no-preview`

選択したプレビューは `dataset_previews` という 1 つの W&B Table にまとめ、`episode`、`camera`、`camera_key` を記録します。episode-camera の組み合わせが 10,000 行を超える場合は upload しません。

プレビュー生成は `wandb.init()` より前にローカルで行います。処理中は対象 episode / camera と進捗を表示し、生成後に実測サイズを確認します。preview budget は `min(250 MiB, canonical dataset directory size の 20%)` です。

budget を超えた場合、対話端末では確認を出し、既定値は No です。CI など非対話環境では `--force-preview-budget` がない限り失敗します。このフラグが許可するのは実測した preview size の超過だけで、10,000 行上限、dataset validation、episode 範囲、encoding error、一時 path の安全性確認、W&B upload error は回避しません。

現行の v3 dataset に対応しています。canonical v2.1 dataset も upload / download / materialize できますが、v2.1 は dataset transfer のみです。`rollout upload` には v3 が必要です。

### 3. 学習用データセットを取得する

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$DATASET_NAME:raw" \
  --root "$TRAIN_DATASET_ROOT"
```

download は途中状態を露出しないように処理します。指定した reference を解決し、Artifact を取得して検証した後で `$TRAIN_DATASET_ROOT` に配置します。ダウンロードが終われば、学習時は LeRobot がそのローカルディレクトリを直接読むため、W&B への接続は不要です。

alias を指定した場合は、コマンドが表示した resolved `vN` reference を保存してください。

### 4. LeRobot で学習する

```bash
lerobot-train \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$TRAIN_DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="$TRAIN_OUTPUT" \
  --job_name="$POLICY_NAME" \
  --batch_size=8 \
  --steps=100000 \
  --policy.push_to_hub=false
```

ここは通常の LeRobot training です。`lerobot-wandb` は `lerobot-train` を wrap せず、学習完了時に model を自動公開することもありません。

保存済みの training config から再開する場合:

```bash
lerobot-train --resume=true \
  --config_path="$POLICY_ROOT/train_config.json"
```

次へ進む前に checkpoint の実際の構成を確認してください。PEFT / LoRA の adapter-only directory も Artifact として保存できますが、rollout や Registry で使うには merged checkpoint など self-contained な policy が必要です。

### 5. 学習済みモデルを公開する

```bash
lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

upload 前には、必要な設定ファイルと重みファイルが揃っているかを確認します。重みのロードや実行は行わないため、ポリシー固有の検証は別途実施してください。

self-contained な model を W&B Registry にも紐付ける場合は `--registry-collection "$POLICY_NAME"` を追加します。adapter-only など不完全な model も Artifact として保存できますが、deployable な Registry entry として扱うべきではありません。

### 6. 評価し、ロールアウト結果を残す

robot machine に candidate model を取得します。

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$POLICY_NAME:candidate" \
  --root "$DOWNLOADED_POLICY_ROOT"
```

evaluation lineage には、alias ではなくコマンドが表示した immutable reference を使います。`v0` と決め打ちしないでください。

```bash
export MODEL_REF="paste-the-resolved-vN-reference-here"
```

LeRobot で policy を実行します。

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="$DOWNLOADED_POLICY_ROOT" \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --dataset.repo_id="local/$ROLLOUT_NAME" \
  --dataset.root="$ROLLOUT_ROOT" \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.push_to_hub=false
```

評価後、成功 episode 数を指定して rollout dataset を upload します。

```bash
export EPISODES_SUCCEEDED="14"

lerobot-wandb rollout upload \
  --root "$ROLLOUT_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$ROLLOUT_NAME" \
  --model-ref "$MODEL_REF" \
  --episodes-succeeded "$EPISODES_SUCCEEDED"
```

成功数は評価者が入力します。`lerobot-wandb` 自身が物理タスクの成功／失敗を推定するわけではありません。rollout は独立した `rollout` Artifact として保存し、評価した model を lineage input として記録します。canonical rollout video は変更せず、動画がある場合だけブラウザ再生用の H.264/yuv420p 派生ファイルを Run Media に記録します。

### 7. 評価済みモデルを昇格する

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection "$POLICY_NAME"
```

promote は alias を付け替え、必要に応じて Registry link を追加します。model bytes の再 upload は行いません。評価に使った exact version をそのまま昇格してください。

## 保存されるもの

| 対象 | 保存先 |
| --- | --- |
| 教示データ | local dataset → W&B `dataset` Artifact |
| 学習入力 | `$TRAIN_DATASET_ROOT` の local directory |
| 学習済み policy | local checkpoint → W&B `model` Artifact |
| 評価 episode | local rollout directory → W&B `rollout` Artifact |
| dataset → model の記録 | dataset reference と local training config |
| model → rollout の記録 | rollout Run input edge と rollout Artifact metadata |

## この companion が行わないこと

以前の LeRobot fork には training path 内部へ W&B を組み込む処理がありました。このリポジトリでは、その hook を意図的に再実装していません。

具体的には、次のことは行いません。

- `lerobot-train` の引数として W&B Artifact reference を直接受け取る
- training command 内部で dataset を materialize する
- 同じ training Run で final model を自動公開する
- `lerobot-record`、`lerobot-train`、`lerobot-rollout` を置き換える
- LeRobot を monkey-patch する、または `lerobot` package 内へ file を追加する
- streaming recorder や deployment controller として動作する

役割をここまでに限定することで、通常の upstream LeRobot 環境の横に追加して使えます。

## トラブルシューティング

- **LeRobot が見つからない／version が対象外:** LeRobot を入れた environment を有効化し、upstream LeRobot `0.6.1` が入っているか確認してください。`--allow-unsupported-lerobot` は、自分で互換性を確認した場合だけ使います。
- **`lerobot-wandb` と `lerobot-*` が別 environment を参照する:** LeRobot environment を有効化し直し、その中で `uv pip install` してください。
- **preview encoding が使えない:** `dataset upload` に `--no-preview` を付けます。canonical Artifact file には影響しません。
- **preview size が budget を超える:** 選択 episode を減らす、`--no-preview` を使う、または非対話環境で意図した超過なら `--force-preview-budget` を指定します。
- **alias の参照先が変わった:** training record、rollout lineage、promote には resolved `vN` reference を使ってください。
- **Registry に link できない:** adapter-only ではなく、deploy 可能な self-contained policy が揃っているか確認してください。
- **README と実際の command が異なる:** インストール済み release では `lerobot-wandb --help`、各 subcommand の `--help`、`pyproject.toml` を command reference としてください。

## 開発

repository root で次を実行します。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

LeRobot がない環境では LeRobot 依存テストを skip します。compatibility CI は対応する upstream release を入れてからテストします。build check では wheel が `lerobot_wandb/*` と `lerobot-wandb` console entry point だけを所有していることも確認します。

## 移行履歴

この repository は [guchengwei/lerobot](https://github.com/guchengwei/lerobot) の source commit `ebdc227057056e077f90fa10155fd505fa53989d` から、[issue #46](https://github.com/guchengwei/lerobot/issues/46) に基づいて切り出しました。元 fork から持ってきた範囲と、意図的に残した fork 固有機能は [MIGRATION.md](./MIGRATION.md) に記載しています。

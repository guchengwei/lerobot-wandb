# lerobot-wandb

[English](./README.md)

`lerobot-wandb` は、LeRobot で作成したデータセット、学習済みポリシー、ロールアウト結果を W&B Artifacts で管理するための補助 CLI です。upstream [LeRobot](https://github.com/huggingface/lerobot) 自体には手を加えません。

![LeRobot と W&B のワークフロー](./assets/wandb-workflow-overview-ja.jpg)

この図は LeRobot と W&B を組み合わせた全体の流れを示しています。`lerobot-wandb` が担当するのは、記録・学習・ロールアウトの前後で Artifact を扱う部分です。ロボットを動かすコマンドは LeRobot の標準コマンドをそのまま使います。

## 役割

`lerobot-wandb` は、既存の LeRobot 環境に追加して使います。LeRobot の処理をラップするのではなく、ローカルディレクトリと W&B Artifact の間をつなぎます。

主な機能は次のとおりです。

- データセット、モデル、ロールアウトデータセットを W&B Artifacts としてアップロード／ダウンロードする
- アップロード前とダウンロード後にローカルディレクトリを検証する
- 元の動画を変更せず、ブラウザ確認用のプレビューを作成する
- 指定した Artifact 参照と、実際に解決された不変のバージョン参照を記録する
- ロールアウト結果を、評価に使ったモデルバージョンへ紐付ける
- 評価済みモデルを alias または W&B Registry で昇格する

パッケージの所有範囲は明確に分けています。`lerobot-wandb` は LeRobot 本体とは別の Python パッケージであり、LeRobot のネイティブプラグインではありません。

- 配布パッケージ: `lerobot-wandb`
- Python パッケージ: `lerobot_wandb`
- CLI コマンド: `lerobot-wandb`

ロボット制御、記録、学習、ポリシー実行は引き続き LeRobot が担当します。

| 作業 | コマンド | 担当 |
| --- | --- | --- |
| 教示データの記録 | `lerobot-record` | LeRobot |
| データセットのアップロード／ダウンロード | `lerobot-wandb dataset ...` | `lerobot-wandb` |
| ポリシー学習 | `lerobot-train` | LeRobot |
| モデルのアップロード／ダウンロード／昇格 | `lerobot-wandb model ...` | `lerobot-wandb` |
| ロボット上でのポリシー実行 | `lerobot-rollout` | LeRobot |
| ロールアウト結果の公開 | `lerobot-wandb rollout upload` | `lerobot-wandb` |

両者の受け渡しはローカルディレクトリを介して行います。W&B が関与するのは完成した Artifact の保存・取得だけで、ロボットの制御ループには入りません。

## 動作条件

- Python 3.12 以降
- W&B アカウント、および Artifact の送受信時に利用できるネットワーク接続
- LeRobot のデータセットや動画を扱うコマンドでは upstream LeRobot `>=0.6.1,<0.6.2`
- 利用するロボット、カメラ、動画処理に必要な LeRobot 側の依存関係

現在のパッケージバージョンは `0.1.0` です。まだ PyPI には公開していないため、GitHub からインストールします。

## インストール

`lerobot-wandb` は LeRobot と同じ Python 環境にインストールしてください。LeRobot のデータを扱うコマンドは、その環境にある LeRobot を import します。

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

LeRobot を base package の hard dependency にしていないのは、依存解決によって既存の LeRobot が意図せず入れ替わるのを避けるためです。LeRobot を必要とするコマンドは実行時にバージョンを確認します。`--allow-unsupported-lerobot` は対応バージョンのチェックを回避するための実験用オプションであり、互換性を保証するものではありません。

## アンインストール

LeRobot と同じ環境を有効化し、`lerobot-wandb` だけを削除します。

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip uninstall lerobot-wandb
```

削除されるのは `lerobot-wandb` の配布パッケージ、`lerobot_wandb` Python パッケージ、`lerobot-wandb` CLI コマンドです。LeRobot 自体は削除も変更もされません。`wandb`、`datasets`、`pandas` などの共有依存パッケージも自動では削除しません。

アンインストールしても、ローカルのデータセット、ダウンロード／materialize 済み Artifact、モデル、ロールアウトディレクトリ、学習出力、サイドカーメタデータなどのユーザーデータは残ります。W&B 上の Artifact、Run、Registry object も削除しません。W&B の認証情報や設定もそのまま残ります。

## ワークフロー例の設定

以下のコマンドでは同じ名前とパスを繰り返し使うため、最初に環境変数へまとめておきます。

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

コマンド例は Linux と Bash を前提にしています。Windows では同じ LeRobot 環境を PowerShell で有効化し、`/dev/ttyACM*` を実際の `COM` ポートに置き換えてください。

## 最短手順: W&B のデータセットから学習する

すでに W&B にデータセットがある場合は、ダウンロード → 学習 → モデルのアップロードの 3 段階です。

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
  --job_name="$POLICY_NAME" \
  --steps=100000 \
  --wandb.enable=true \
  --wandb.entity="$WANDB_ENTITY" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false

lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

学習 Run とメトリクスは LeRobot 標準の W&B 連携で記録します。LeRobot 0.6.1 には checkpoint を W&B Artifact として保存する機能もありますが、通常の checkpoint Artifact は後の手順で使う完全なポリシーディレクトリではありません。この例ではその保存を無効にし、代わりに `$POLICY_ROOT` 全体を `lerobot-wandb model upload` で公開します。

checkpoint の配置はポリシーと学習設定で変わります。アップロード前に LeRobot の実際の出力先を確認してください。

## 一連の流れ

Artifact 側の手順はロボット機種に依存しません。以下では、ロボットを扱うコマンドだけ具体例として SO-101 を使います。

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

`repo_id` は LeRobot 側のローカル識別子です。`root` には、後で `lerobot-wandb` が検証してアップロードするディレクトリを指定します。

### 2. データセットを W&B に公開する

```bash
lerobot-wandb dataset upload \
  --root "$DATASET_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$DATASET_NAME" \
  --alias raw
```

W&B Run を開始する前にデータセットを検証します。アップロードが完了すると、次のような変更されないバージョン参照が表示されます。

```text
your-wandb-entity/your-wandb-project/your-dataset:v0
```

再現性が必要な記録には alias ではなく `vN` を残してください。`raw` のような alias は後から別バージョンを指すことがあります。

#### データセット動画のプレビュー

元の動画は変更せず、そのまま Artifact に保存します。ブラウザ確認用のプレビューだけを別ファイルとして作成します。

対象の選び方:

- 既定: 各カメラから決定的に選んだ代表エピソードを 1 件
- 指定したエピソード: `--preview-episode INDEX` を必要な回数だけ指定
- 全エピソード／全カメラ: `--preview-all`
- プレビューなし: `--no-preview`

選択したプレビューは `dataset_previews` という 1 つの W&B Table にまとめ、`episode`、`camera`、`camera_key` を記録します。エピソードとカメラの組み合わせが 10,000 行を超える場合はアップロードしません。

プレビュー生成は `wandb.init()` より前にローカルで行います。処理中は対象のエピソード／カメラと進捗を表示し、生成後に実測サイズを確認します。容量上限は `min(250 MiB, 元データセットディレクトリ容量の 20%)` です。

実測サイズが上限を超えた場合、対話端末では確認を表示し、既定値は No です。CI などの非対話環境では `--force-preview-budget` がない限り失敗します。このフラグが許可するのは実測したプレビュー容量の超過だけで、10,000 行上限、データセット検証、エピソード範囲、エンコードエラー、一時パスの安全性確認、W&B アップロードエラーは回避しません。

現行の v3 データセットに対応しています。canonical v2.1 データセットもアップロード、ダウンロード、materialize できますが、v2.1 で対応するのはデータセット転送だけです。`rollout upload` には v3 が必要です。

### 3. 学習用データセットを取得する

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$DATASET_NAME:raw" \
  --root "$TRAIN_DATASET_ROOT"
```

ダウンロードは途中状態を公開しないように処理します。指定した参照を解決し、Artifact を取得して検証した後で `$TRAIN_DATASET_ROOT` に配置します。ダウンロード完了後、LeRobot はそのローカルディレクトリを直接読み込むため、学習データを読むために W&B 接続は必要ありません。ただし、下記の `--wandb.enable=true` で学習メトリクスを同期する場合は、学習中も W&B への接続が必要です。

alias を指定した場合は、コマンドが表示した解決済みの `vN` 参照を保存してください。

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
  --wandb.enable=true \
  --wandb.entity="$WANDB_ENTITY" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false
```

ここは通常の LeRobot の学習処理です。`--wandb.*` は LeRobot 標準の W&B 連携で、学習 Run とメトリクスを `$WANDB_ENTITY/$WANDB_PROJECT` に記録します。LeRobot 0.6.1 は checkpoint Artifact も W&B にアップロードできます。ただし通常の checkpoint Artifact に含まれるのは重みだけで、`config.json` を含む完全なポリシーディレクトリではありません。そこでこの例では `--wandb.disable_artifact=true` でそのアップロードを止め、次の手順で `$POLICY_ROOT` 全体を `lerobot-wandb model upload` に渡します。以後のダウンロード、ロールアウトの lineage、alias、Registry への登録には、こちらの Artifact を使います。`lerobot-wandb` は `lerobot-train` をラップせず、学習完了時にモデルを自動公開することもありません。

保存済みの学習設定から再開する場合:

```bash
lerobot-train --resume=true \
  --config_path="$POLICY_ROOT/train_config.json"
```

次へ進む前に checkpoint の実際の構成を確認してください。PEFT / LoRA の adapter-only ディレクトリも Artifact として保存できますが、ロールアウトや Registry で使うには merged checkpoint など自己完結したポリシーが必要です。

### 5. 学習済みモデルを公開する

```bash
lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

アップロード前には、必要な設定ファイルと重みファイルが揃っているかを確認します。重みをロードしたり実行したりはしません。ポリシー固有の検証は別途実施してください。

自己完結したモデルを W&B Registry にも紐付ける場合は `--registry-collection "$POLICY_NAME"` を追加します。adapter-only など不完全なモデルも Artifact として保存できますが、デプロイ可能な Registry entry として扱うべきではありません。

### 6. 評価し、ロールアウト結果を残す

ロボットを動かすマシンに候補モデルを取得します。

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$POLICY_NAME:candidate" \
  --root "$DOWNLOADED_POLICY_ROOT"
```

評価の lineage には、alias ではなくコマンドが表示した変更されないバージョン参照を使います。`v0` と決め打ちしないでください。

```bash
export MODEL_REF="paste-the-resolved-vN-reference-here"
```

LeRobot でポリシーを実行します。

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

評価後、成功エピソード数を指定してロールアウトデータセットをアップロードします。

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

成功数は評価者が入力します。`lerobot-wandb` 自身が物理タスクの成功／失敗を推定するわけではありません。ロールアウトは独立した `rollout` Artifact として保存し、評価したモデルを lineage input として記録します。元のロールアウト動画は変更せず、動画がある場合だけブラウザ再生用の H.264/yuv420p 派生ファイルを Run Media に記録します。

### 7. 評価済みモデルを昇格する

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection "$POLICY_NAME"
```

昇格では alias を付け替え、必要に応じて Registry link を追加します。モデル本体を再アップロードすることはありません。評価に使った正確なバージョンをそのまま昇格してください。

## 保存されるもの

| 対象 | 保存先 |
| --- | --- |
| 教示データ | ローカルデータセット → W&B `dataset` Artifact |
| 学習入力 | `$TRAIN_DATASET_ROOT` のローカルディレクトリ |
| 学習メトリクス | `lerobot-train` が `$WANDB_PROJECT` に作成する W&B Run |
| 学習済みポリシー | ローカル checkpoint → W&B `model` Artifact |
| 評価エピソード | ローカルのロールアウトディレクトリ → W&B `rollout` Artifact |
| データセットからモデルまでの記録 | データセット参照とローカル学習設定 |
| モデルからロールアウトまでの記録 | rollout Run input edge と rollout Artifact metadata |

## このツールが行わないこと

以前の LeRobot fork には、学習処理の内部へ companion 固有の W&B Artifact 処理を組み込む機能がありました。このリポジトリでは、そのフックは再実装していません。upstream LeRobot 標準の `--wandb.*` による学習ログはそのまま利用できます。

具体的には、次のことは行いません。

- `lerobot-train` の引数として W&B Artifact 参照を直接受け取る
- 学習コマンドの内部でデータセットを materialize する
- 同じ training Run で final model を自動公開する
- `lerobot-record`、`lerobot-train`、`lerobot-rollout` を置き換える
- LeRobot を monkey-patch する、または `lerobot` パッケージ内へファイルを追加する
- ストリーミングレコーダーやデプロイメントコントローラーとして動作する

役割をここまでに限定することで、通常の upstream LeRobot 環境へ安全に追加できます。

## トラブルシューティング

- **LeRobot が見つからない／バージョンが対象外:** LeRobot を入れた環境を有効化し、upstream LeRobot `0.6.1` が入っているか確認してください。`--allow-unsupported-lerobot` は、自分で互換性を確認した場合だけ使います。
- **`lerobot-wandb` と `lerobot-*` が別の環境を参照する:** LeRobot の環境を有効化し直し、その中で `uv pip install` してください。
- **プレビューのエンコードが使えない:** `dataset upload` に `--no-preview` を付けます。元の Artifact ファイルには影響しません。
- **プレビュー容量が上限を超える:** 対象エピソードを減らす、`--no-preview` を使う、または非対話環境で意図した超過なら `--force-preview-budget` を指定します。
- **alias の参照先が変わった:** 学習記録、ロールアウトの lineage、昇格には解決済みの `vN` 参照を使ってください。
- **Registry に link できない:** adapter-only ではなく、デプロイ可能な自己完結したポリシーが揃っているか確認してください。
- **README と実際のコマンドが異なる:** インストール済み release では `lerobot-wandb --help`、各 subcommand の `--help`、`pyproject.toml` をコマンド仕様として確認してください。

## 開発

リポジトリのルートで次を実行します。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

LeRobot がない環境では LeRobot 依存テストを skip します。compatibility CI は対応する upstream release をインストールしてからテストします。build check では wheel が `lerobot_wandb/*` と `lerobot-wandb` console entry point だけを所有していることも確認します。

## 移行履歴

このリポジトリは [guchengwei/lerobot](https://github.com/guchengwei/lerobot) の source commit `ebdc227057056e077f90fa10155fd505fa53989d` から、[issue #46](https://github.com/guchengwei/lerobot/issues/46) に基づいて切り出しました。元 fork から持ってきた範囲と、意図的に残した fork 固有機能は [MIGRATION.md](./MIGRATION.md) に記載しています。
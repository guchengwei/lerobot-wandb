# lerobot-wandb

[English](./README.md)

既存の [LeRobot](https://github.com/huggingface/lerobot) を変更せず、データセット、学習済みポリシー、ロールアウト結果を Weights & Biases Artifacts で管理するための companion CLI です。

![LeRobot と W&B のワークフロー](./assets/wandb-workflow-overview-ja.jpg)

上の図は LeRobot と W&B を組み合わせた全体の流れを示しています。`lerobot-wandb` が担当するのは、記録・学習・ロールアウトの前後にある Artifact 操作です。ロボットを動かすコマンドには通常の LeRobot を使います。

## このパッケージでできること

`lerobot-wandb` は、既存の LeRobot 環境に追加して使います。ローカルの LeRobot ディレクトリを W&B 上でバージョン管理し、次の操作を行えます。

- データセット、モデル、ロールアウトデータセットを W&B Artifacts としてアップロード・ダウンロードする
- アップロード前とダウンロード後にローカルディレクトリを検証する
- 元の動画を変更せず、ブラウザで再生できるデータセット／ロールアウトのプレビューを作る
- 指定した Artifact と実際に解決されたバージョンを記録する
- ロールアウト結果を、その評価で使った正確なモデルバージョンに関連付ける
- 評価済みモデルを alias または W&B Registry で昇格する

これは LeRobot 本体とは別のパッケージであり、LeRobot のネイティブプラグインではありません。

- 配布名: `lerobot-wandb`
- Python パッケージ: `lerobot_wandb`
- コマンド: `lerobot-wandb`

## LeRobot との役割分担

| 作業 | コマンド | 担当 |
| --- | --- | --- |
| 教示データを記録する | `lerobot-record` | upstream LeRobot |
| データセットをアップロード／ダウンロードする | `lerobot-wandb dataset ...` | この companion |
| ポリシーを学習する | `lerobot-train` | upstream LeRobot |
| ポリシーをアップロード／ダウンロード／昇格する | `lerobot-wandb model ...` | この companion |
| ロボットでポリシーを実行する | `lerobot-rollout` | upstream LeRobot |
| ロールアウト結果を公開する | `lerobot-wandb rollout upload` | この companion |

両者の受け渡しにはローカルディレクトリを使います。W&B は完成した Artifact の保存先であり、ロボットの制御ループには入りません。

## 動作条件

- Python 3.12 以降
- W&B アカウントと、Artifact 操作時のネットワーク接続
- LeRobot のデータセットや動画を扱うコマンドでは upstream LeRobot `>=0.6.1,<0.6.2`
- 使用するロボット、カメラ、動画処理に必要な通常の LeRobot 依存関係

現在のパッケージバージョンは `0.1.0` です。まだ PyPI には公開されていないため、GitHub からインストールします。

## インストール

`lerobot-wandb` は **LeRobot と同じ Python 環境**にインストールしてください。LeRobot に依存する companion コマンドは、有効な環境に入っている LeRobot を import します。

LeRobot の checkout が `.venv` を使っている場合の例:

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
wandb login
lerobot-wandb --help
```

新しくセットアップする場合は、LeRobot の `uv` 環境構成に合わせ、1 つの `.venv` に両方のパッケージをインストールします。

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

`feetech` は、以下の SO-101 例で使うモーター依存関係です。別のロボットでは、対応する extra に置き換えてください。FFmpeg などのシステムパッケージと、環境別の PyTorch 手順は upstream の [LeRobot インストールガイド](https://huggingface.co/docs/lerobot/installation) に従ってください。

基本パッケージでは LeRobot を hard dependency にしていません。依存関係の解決時に、既存の upstream LeRobot を別バージョンへ置き換えないためです。LeRobot を必要とするコマンドは、実行時にインストール済みバージョンを確認します。`--allow-unsupported-lerobot` は実験用の回避オプションであり、互換性を保証するものではありません。

## アンインストール

インストール時と同じ LeRobot 環境を有効化し、パッケージマネージャーで companion だけを削除します。

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip uninstall lerobot-wandb
```

削除されるのは `lerobot-wandb` distribution、`lerobot_wandb` import package、`lerobot-wandb` console command だけです。LeRobot はインストールされたままで、変更されません。`wandb`、`datasets`、`pandas` などの共有依存パッケージも自動では削除しません。

アンインストールしても、ローカルの dataset、download または materialize 済み Artifact、model、rollout directory、training output、sidecar metadata などのユーザーデータは削除しません。また、W&B 上の Artifact、Run、Registry object は削除しません。W&B の認証情報や設定も削除しません。

## ワークフローの設定例

以下の例で使う W&B 上の名前を先に設定し、その後の手順では同じ変数を使います。

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

これらは companion が要求する固定名ではありません。自分のプロジェクト用に一度設定すれば、以下のコマンドでそのまま再利用できます。

コマンド例は Linux と Bash 互換シェルを前提にしています。Windows では同じ LeRobot 環境を PowerShell 用の activation コマンドで有効化し、`/dev/ttyACM*` を対応する `COM` ポートに置き換えてください。

## クイックスタート: W&B のデータセットから学習する

データセットがすでに W&B にある場合は、次の手順が最短です。

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

checkpoint の場所は、ポリシーと学習設定によって変わります。アップロード前に LeRobot の学習結果で実際のパスを確認してください。

## 一連のワークフロー

Artifact の流れ自体は SO-101 固有ではありません。以下の robot-facing コマンドでは LeRobot の具体例として SO-101 を使っています。別のロボットでは、robot、teleoperator、camera、task、ローカルパスの引数を置き換えれば、companion 側の手順は同じです。

### 1. 教示データをローカルに記録する

記録には upstream LeRobot の通常のコマンドを使います。この時点では W&B は使いません。

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

`repo_id` は LeRobot が使うローカルのラベルです。`root` には、companion が検証してアップロードするディレクトリを指定します。

### 2. データセットをアップロードする

```bash
lerobot-wandb dataset upload \
  --root "$DATASET_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$DATASET_NAME" \
  --alias raw
```

コマンドは W&B Run を作成する前にデータセットを検証し、次のような変更されない参照先を表示します。

```text
your-wandb-entity/your-wandb-project/your-dataset:v0
```

再現性が必要なときは、この `vN` 形式の参照先を保存してください。`raw` のような alias は、後から別バージョンを指す場合があります。

#### 動画プレビュー

元の動画ファイルは変更せず、そのまま Artifact に保存します。ブラウザ用プレビューは、レビューのために別途作られる再生可能な派生ファイルです。

- 既定: 各カメラから決定的に選ばれた代表エピソード 1 件
- 指定したエピソード: `--preview-episode INDEX` を必要な回数だけ指定
- 全エピソード／全カメラ: `--preview-all`
- プレビューを無効化: `--no-preview`

Run は、選択したレビュー用メディアを、選択したエピソード／カメラの組み合わせごとに 1 行を持つ安定した `dataset_previews` Table 1 つに記録します。この Table では `episode`、`camera`、完全な `camera_key` を使ってフィルターとグループ化ができ、数値型の `episode` は数値順にソートできます。例:

```text
episode = 12
camera = front
camera_key = observation.images.front
```

選択結果が 10,000 エピソード／カメラ行を超える場合、CLI は対話型環境でも非対話型環境でも必ず拒否します。明示的な `--preview-episode` の指定数を減らすか、`--no-preview` を使ってください。

プレビューの準備は `wandb.init()` より前にローカルで行われるため、ファイルの準備中に W&B Run は作成されません。CLI はすぐにバッチ件数を表示し、選択した各エピソード／カメラの開始と完了を表示します（例: `[1/4] episode 12 · observation.images.front`）。トランスコードの再生時間が分かる場合は割合を表示し、分からない場合は割合や ETA を推測せず、処理中またはフレーム数を表示します。準備が終わると合計容量を表示し、`Starting W&B upload...` と通知します。

容量上限の式は変わらず、`min(250 MiB, canonical データセットディレクトリのバイト数の 20%)` です。選択したプレビューを準備して容量を実測した後、その値を上限と比較します。プレビューは確認用であり、学習データではありません。実測したプレビュー容量が上限を超えた場合、対話型端末では両方の値を表示して `[y/N]`（既定値は **No**）で確認します。`yes` なら準備済みファイルをそのまま使って続行し、`no`、EOF、その他の入力なら `wandb.init()` より前に中止します。

非対話または CI 環境では、`--force-preview-budget` を指定しない限り、容量超過を W&B 開始前にエラーにします。上限内ではこのフラグは何もせず、上限超過時も実測したプレビュー容量の超過だけを許可します。10,000 行の Table 上限、データセット／スキーマ検証、エピソード選択範囲の検証、エンコード失敗、一時パスの安全性チェック、W&B アップロードエラーを回避するものではありません。超過を避けるには、`--preview-all` の代わりに `--preview-episode` の指定を減らすなど選択範囲を小さくするか、`--no-preview` を使ってください。

現行の v3 データセットに対応しています。canonical v2.1 データセットもアップロード、ダウンロード、ローカル配置ができます。ただし、v2.1 で対応するのはデータセット転送までです。`rollout upload` には v3 が必要です。

### 3. データセットをダウンロードしてローカルに配置する

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$DATASET_NAME:raw" \
  --root "$TRAIN_DATASET_ROOT"
```

指定した参照先を解決し、Artifact をトランザクションとしてダウンロードし、内容を検証して `$TRAIN_DATASET_ROOT` に配置します。ダウンロードが終われば、学習時に LeRobot がこのローカルディレクトリを直接読むため、W&B への接続は不要です。

alias を指定した場合は、コマンドが表示する解決済みの `vN` 参照先を記録してください。

### 4. Upstream LeRobot で学習する

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

これは通常の LeRobot 学習です。companion はコマンドをラップせず、学習完了時にモデルを自動公開することもありません。

保存済みの学習設定から再開する場合:

```bash
lerobot-train --resume=true \
  --config_path="$POLICY_ROOT/train_config.json"
```

次の手順へ進む前に、実際の checkpoint 構成を確認してください。PEFT／LoRA の adapter-only ディレクトリも Artifact として保存できますが、ロールアウトや Registry には merged checkpoint などの self-contained なポリシーが必要です。

### 5. 学習済みモデルをアップロードする

```bash
lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

アップロード前の検証では、必要な設定ファイルと重みファイルの有無を確認します。重みのロードや実行は行わないため、ポリシー固有の検証は別途実施してください。

self-contained なモデルを W&B Registry にも登録する場合は、`--registry-collection "$POLICY_NAME"` を追加します。デプロイできないモデルも Artifact として保存できますが、deployable Registry link は作成できません。

### 6. モデルをダウンロードし、ロールアウトを公開する

ロボットを動かすマシンに候補モデルをダウンロードします。

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$POLICY_NAME:candidate" \
  --root "$DOWNLOADED_POLICY_ROOT"
```

コマンドが表示する変更されない参照先をコピーします。ロールアウトの lineage には、移動する可能性がある `candidate` alias ではなく、この参照先を使います。正確な `vN` はアップロード結果で決まるため、`v0` と決め打ちしないでください。

```bash
export MODEL_REF="paste-the-resolved-vN-reference-here"
```

Upstream LeRobot でポリシーを実行します。

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

評価中に成功したエピソード数を数えます。ロボットを切り離してから結果を公開します。

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

成功数は operator が入力します。companion が物理タスクの成否を判定することはありません。結果は独立した `rollout` Artifact として保存され、評価したモデルが lineage input として関連付けられます。元のロールアウト動画は変更されません。動画がある場合は、ブラウザ再生用の H.264/yuv420p 派生ファイルを Run Media に記録します。

### 7. 評価済みモデルを昇格する

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection "$POLICY_NAME"
```

昇格では alias を移動し、必要に応じて Registry link を追加します。モデル本体の再アップロードは行いません。評価に使った正確なバージョンを指定してください。ダウンロード済みディレクトリを再アップロードすると、ロールアウトとの lineage edge を持たない新しいモデルバージョンが作られます。

## 作成される W&B オブジェクト

| 対象 | 保存先 |
| --- | --- |
| 教示データ | `dataset` Artifact collection `$DATASET_NAME` |
| 学習入力 | ローカルに配置した `$TRAIN_DATASET_ROOT` |
| 学習済みポリシー | ローカル checkpoint、続いて `model` Artifact `$POLICY_NAME` |
| 評価エピソード | ローカルの rollout tree、続いて `rollout` Artifact `$ROLLOUT_NAME` |
| データセットからポリシーまでの記録 | 使用したデータセット参照先とローカル学習設定 |
| ポリシーからロールアウトまでの記録 | rollout Run の input edge と rollout Artifact metadata |

## この companion が行わないこと

以前の LeRobot fork には、学習コマンドの内部で W&B を扱う機能がありました。このリポジトリでは、その hook を意図的に再実装していません。次の処理は行いません。

- `lerobot-train` に W&B Artifact の参照先を直接渡す
- 学習コマンドの内部でデータセットを materialize する
- 同じ training Run で final model を公開する
- upstream の `lerobot-record`、`lerobot-train`、`lerobot-rollout` を置き換える
- LeRobot を monkey-patch する、または `lerobot` パッケージ内にファイルを追加する
- streaming recorder や deployment controller として動作する

役割を分離することで、通常の upstream LeRobot 環境に companion を追加して使えます。

## トラブルシューティング

- **LeRobot がない、またはバージョンが非対応:** LeRobot が入っている同じ Python 環境を有効化しているか確認し、その環境に upstream LeRobot `0.6.1` が入っていることを確認してください。`--allow-unsupported-lerobot` は、自分の環境で互換性を確認した場合だけ使用します。
- **`lerobot-wandb` と `lerobot-*` が別の環境から実行される:** LeRobot の virtual environment を有効化し直し、その環境で `uv pip install` を使って companion をインストールしてください。LeRobot に依存する操作では、両方のコマンドを同じ環境に置く必要があります。
- **プレビューをエンコードできない:** dataset upload に `--no-preview` を追加してください。元の Artifact ファイルには影響しません。
- **プレビュー容量が上限を超える:** 確認の既定値は No です。`yes` と答えると準備済みファイルを使って続行します。CI など非対話環境では、選択範囲を小さくするか `--no-preview` を使うか、`--force-preview-budget` を付けて再実行してください。このフラグが許可するのは実測した容量超過だけで、他の安全策はそのまま働きます。
- **ダウンロードに使った alias の参照先が変わった:** 学習記録、ロールアウトの lineage、昇格には、コマンドが表示した `vN` 形式の参照先を使ってください。
- **モデルを Registry に登録できない:** adapter-only ではなく、デプロイ可能な完全なポリシーがディレクトリに入っているか確認してください。
- **このページとコマンド／オプションが異なる:** このリリースでは `lerobot-wandb --help`、各 subcommand の `--help`、`pyproject.toml` をコマンド仕様として確認してください。

## 開発

リポジトリのルートで次を実行します。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

LeRobot がない環境では、LeRobot に依存するテストを skip します。互換性 CI では対応する upstream release をインストールしてからテストします。build check では、wheel が `lerobot_wandb/*` と `lerobot-wandb` console entry point だけを所有し、`lerobot/*` を含まないことも確認します。

## 移行履歴

このリポジトリは [guchengwei/lerobot](https://github.com/guchengwei/lerobot) の commit `ebdc227057056e077f90fa10155fd505fa53989d` から、companion の Artifact transfer 機能を独立させた snapshot です。[issue #46](https://github.com/guchengwei/lerobot/issues/46) に基づいて作成されました。移行元の範囲と除外した fork 固有 hook は [MIGRATION.md](./MIGRATION.md) に記録しています。

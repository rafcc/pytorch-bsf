# Contributing Guide

このドキュメントは pytorch-bsf の開発に参加するための手引きです。

## 目次

1. [開発の方針](#開発の方針)
2. [開発環境の構築方法](#開発環境の構築方法)
3. [最初にやってみること](#最初にやってみること)
4. [開発の進め方](#開発の進め方)
5. [コミットのルール](#コミットのルール)
6. [コーディング規約](#コーディング規約)
7. [テストの書き方](#テストの書き方)
8. [フォルダ構成](#フォルダ構成)

---

## 開発の方針

pytorch-bsf はベジエ単体フィッティング（Bézier Simplex Fitting）のための Python ライブラリです。以下の方針に従って開発を進めます。

- **シンプルな API**: ユーザーが `torch_bsf.fit()` の 1 行だけで使えるように、API はシンプルに保つ。
- **後方互換性の維持**: マイナーバージョンアップでは既存の API を壊さない。破壊的変更はメジャーバージョンアップで行う。
- **型安全性**: すべての公開 API に型ヒントを付ける（PEP 561 準拠）。
- **テストカバレッジの維持**: 新機能や修正には必ずテストを追加する。
- **ドキュメントの同期**: 公開 API の変更には Sphinx ドキュメントおよび docstring の更新を伴う。

---

## 開発環境の構築方法

### 前提条件

- Python 3.10 以上
- [Conda](https://docs.conda.io/) または pip

### Conda を使う場合（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/opthub-org/pytorch-bsf.git
cd pytorch-bsf

# Conda 環境を作成・有効化
conda env create -f environment.yml
conda activate pytorch-bsf
```

### pip を使う場合

```bash
# リポジトリをクローン
git clone https://github.com/opthub-org/pytorch-bsf.git
cd pytorch-bsf

# 開発用依存パッケージを含めて editable インストール
pip install -e ".[develop]"
```

開発用依存パッケージ（`[develop]` extras）には以下が含まれます：

| パッケージ | 用途 |
|-----------|------|
| `pytest`, `pytest-cov`, `pytest-randomly` | テスト実行 |
| `mypy` + 型スタブ | 静的型チェック |
| `black` | コードフォーマット |
| `isort` | インポート整列 |
| `tox` | テスト自動化 |

---

## 最初にやってみること

環境構築後、以下を実行して開発環境が正しくセットアップされていることを確認してください。

### 1. テストを実行する

```bash
pytest tests/
```

すべてのテストが PASSED になることを確認します。

### 2. CLI を動かしてみる

リポジトリにはサンプルデータ（`params.csv`, `values.csv`）が含まれています。

```bash
python -m torch_bsf --params=params.csv --values=values.csv --degree=3
```

### 3. クイックスタートスクリプトを実行する

```bash
bash examples/quickstart/run.sh
```

### 4. MLflow で実行する

```bash
mlflow run . --entry-point main \
  -P params=params.csv \
  -P values=values.csv \
  -P degree=3
```

---

## 開発の進め方

### イシューの立て方

- バグ報告には **Bug report** テンプレートを使用してください。
- 機能要望には **Feature request** テンプレートを使用してください。
- イシューには再現手順・期待する動作・実際の動作を明確に記述してください。
- 関連するイシューや PR へのリンクを貼ると議論がスムーズになります。

### ブランチの切り方

`master` ブランチから作業ブランチを切ってください。

```bash
git checkout master
git pull origin master
git checkout -b <ブランチ名>
```

ブランチ名はコミットタイプとイシュー番号、短い説明を組み合わせた形式を推奨します。

| 種別 | 例 |
|------|-----|
| 新機能 | `feat/123-add-new-sampler` |
| バグ修正 | `fix/456-fix-nan-in-output` |
| ドキュメント | `docs/789-update-contributing` |
| リファクタリング | `refactor/101-simplify-validator` |
| CI / ビルド関連 | `ci/102-update-workflow` |

### PR レビューとマージ

1. 作業ブランチへの変更をプッシュし、`master` に向けた Pull Request を作成します。
2. PR の説明欄には変更内容・関連イシュー（`Closes #XXX`）・テスト方法を記載してください。
3. CI（GitHub Actions）がすべて通過していることを確認してください。
4. レビュアーのコメントに対応し、Approve を得たらマージします。
5. マージ後、不要なブランチは削除してください。

---

## コミットのルール

このプロジェクトでは [Conventional Commits](https://www.conventionalcommits.org/) の形式に従います。`release-please` がこの形式をもとに CHANGELOG の生成とバージョン管理を自動化しています。

### 形式

```
<type>(<scope>): <subject>
```

- `<scope>` は省略可能です。
- `<subject>` は英語で書き、命令形・現在形で記述します（例: "add feature" ✓、"added feature" ✗）。
- 1 行目は 72 文字以内に収めてください。

### コミットタイプ

| タイプ | 説明 | バージョンへの影響 |
|--------|------|------------------|
| `feat` | 新機能の追加 | マイナーバージョンアップ |
| `fix` | バグ修正 | パッチバージョンアップ |
| `docs` | ドキュメントのみの変更 | なし |
| `style` | コードの動作に影響しない変更（空白、フォーマットなど） | なし |
| `refactor` | バグ修正でも新機能でもないコード変更 | なし |
| `test` | テストの追加・修正 | なし |
| `chore` | ビルドプロセスや補助ツールの変更 | なし |
| `ci` | CI 設定の変更 | なし |
| `deps` | 依存パッケージの追加・更新 | なし |
| `perf` | パフォーマンス改善 | なし |

### 破壊的変更（BREAKING CHANGE）

後方互換性を壊す変更には、コミット本文またはフッターに `BREAKING CHANGE:` を記述します。これによりメジャーバージョンアップが自動的にトリガーされます。

```
feat!: remove deprecated normalize parameter

BREAKING CHANGE: The `normalize` parameter has been removed.
Use `preprocessing` instead.
```

### コミット例

```
feat(model_selection): add elastic net grid search
fix: resolve UnpicklingError in PyTorch 2.6+
docs: add CONTRIBUTING.md
deps: update starlette constraint for mlflow compatibility
test: add parametrized tests for BezierSimplex
ci: add Python 3.14 to test matrix
```

---

## コーディング規約

### フォーマッター

コードを変更したら、以下のツールで整形してからコミットしてください。

```bash
# インポートの整列
isort torch_bsf/ tests/

# コードフォーマット（flake8 の最大行長に合わせて 127 文字を指定）
black --line-length 127 torch_bsf/ tests/
```

### リンター

CI では flake8 によるリントが実行されます。以下のコマンドでローカルでも確認できます。

```bash
# 構文エラーおよび未定義名のチェック（CI と同じ設定）
flake8 --select=E9,F63,F7,F82 --show-source --statistics torch_bsf/ tests/

# 一般的なコードスタイルのチェック
flake8 --max-complexity=10 --max-line-length=127 torch_bsf/ tests/
```

### 型チェック

公開 API にはすべて型ヒントを付けてください。型チェックは mypy で行います。

```bash
mypy torch_bsf/
```

設定（`setup.cfg` の `[mypy]` セクション）：

- `python_version = 3.10`（サポートする最低バージョンを基準として型チェックを行います）
- `warn_return_any = True`
- `warn_unused_configs = True`

### スタイルガイドライン

- **インデント**: スペース 4 個
- **最大行長**: 127 文字（flake8 の `--max-line-length=127` に合わせています。black も `--line-length 127` で実行してください）
- **文字列**: ダブルクォート（black デフォルト）
- **docstring**: Google スタイル推奨
- **型ヒント**: `from __future__ import annotations` を使わず、Python 3.10 の組み込み型ヒント構文を使用

---

## テストの書き方

テストは `tests/` ディレクトリに配置します。

### テストの実行

```bash
# 全テストを実行
pytest tests/

# カバレッジレポートを生成しながら実行
pytest tests/ --cov=torch_bsf --cov-report=html

# 特定のファイルのみ実行
pytest tests/bezier_simplex.py
```

### テストファイルの命名規則

- テストファイル名はテスト対象のモジュール名と合わせます（例: `torch_bsf/bezier_simplex.py` → `tests/bezier_simplex.py`）。
- テスト関数名は `test_` または `_test_` で始めます。

### テストの書き方

pytest の `@pytest.mark.parametrize` を活用して、様々な入力パターンを網羅してください。

```python
import pytest
import torch_bsf as tbsf


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    [
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ],
)
def test_zeros(n_params: int, n_values: int, degree: int) -> None:
    bs = tbsf.BezierSimplex.zeros(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    assert bs.degree == degree
```

### docstring テスト

公開 API の docstring には実行可能な例（doctest）を書いてください。CI で自動的に検証されます。

```python
def fit(params, values, degree):
    """Fit a Bézier simplex to the given data.

    Example:
        >>> import torch
        >>> import torch_bsf
        >>> params = torch.tensor([[0.0], [0.5], [1.0]])
        >>> values = torch.tensor([[0.0], [0.25], [1.0]])
        >>> bs = torch_bsf.fit(params=params, values=values, degree=2)
    """
```

### Sphinx ドキュメントのテスト

`docs/` 内の RST ファイルのコードブロックも Sphinx doctest でテストされます。

```bash
sphinx-build -b doctest docs/ docs/_build/doctest
```

---

## フォルダ構成

```
pytorch-bsf/
├── .github/
│   ├── workflows/               # GitHub Actions ワークフロー
│   │   ├── python-package.yml   # pytest + flake8 + doctest
│   │   ├── release-please-action.yml  # 自動リリース（CHANGELOG 生成・PyPI 公開）
│   │   ├── sphinx-pages.yml     # Sphinx ドキュメントのビルド・公開
│   │   ├── codeql-analysis.yml  # セキュリティスキャン
│   │   └── python-package-conda.yml  # Conda 環境での E2E テスト
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md        # バグ報告テンプレート
│   │   └── feature_request.md   # 機能要望テンプレート
│   ├── CODEOWNERS               # コードオーナー設定
│   └── dependabot.yml           # 依存パッケージ自動更新
├── torch_bsf/                   # メインパッケージ
│   ├── __init__.py              # 公開 API のエクスポート（BezierSimplex, fit）
│   ├── __main__.py              # CLI エントリポイント
│   ├── bezier_simplex.py        # BezierSimplex クラス・fit 関数・DataModule
│   ├── control_points.py        # 制御点の管理・インデックス計算
│   ├── preprocessing.py         # データスケーリング（MinMax, Std, Quantile, None）
│   ├── sampling.py              # サンプリングユーティリティ
│   ├── validator.py             # 入力検証
│   ├── py.typed                 # PEP 561 型ヒントマーカー
│   └── model_selection/         # モデル選択サブパッケージ
│       ├── __init__.py
│       ├── kfold.py             # K 分割交差検証 CLI
│       └── elastic_net_grid.py  # Elastic Net グリッドサーチ CLI
├── tests/                       # テストスイート
│   ├── __init__.py
│   ├── bezier_simplex.py        # BezierSimplex のユニットテスト
│   ├── control_points.py        # ControlPoints のユニットテスト
│   ├── validator.py             # Validator のユニットテスト
│   ├── test_control_point_index_format.py
│   └── data/                    # テスト用データファイル
├── docs/                        # Sphinx ドキュメント
│   ├── conf.py                  # Sphinx 設定
│   ├── index.rst                # ドキュメントトップページ
│   ├── requirements.txt         # ドキュメントビルド用依存パッケージ
│   └── applications/            # アプリケーション例（RST）
├── examples/                    # サンプルスクリプト
│   └── quickstart/
│       └── run.sh               # クイックスタート用シェルスクリプト
├── CHANGELOG.md                 # 変更履歴（release-please が自動生成）
├── CONTRIBUTING.md              # このファイル
├── LICENSE                      # MIT ライセンス
├── README.md                    # プロジェクト概要・使い方
├── MLproject                    # MLflow プロジェクト定義
├── environment.yml              # Conda 環境定義
├── setup.cfg                    # パッケージ設定・依存関係・ツール設定
└── setup.py                     # setuptools エントリポイント（最小限）
```

# Construction Drawing Intelligence

建設図面・技術資料を自然言語で検索し、回答根拠を原本ページまで追跡できる Grounded RAG プロトタイプです。

## 概要

Construction Drawing Intelligence は、建設図面 PDF や Markdown 形式の技術資料を取り込み、自然言語で検索・質問できるローカル実行型システムです。

単に AI が回答を生成するのではなく、回答に使った Evidence をサーバー側で管理し、Citation から元のページや Markdown 節へ戻れるようにしています。PDF では正規化 bbox を使って該当箇所をハイライトできます。

この公開リポジトリに含まれるデータは、すべて公開用に作成した架空のサンプルです。実在する企業・建物・工事・契約の情報は含みません。

## 主な機能

- PDF / Markdown の取り込み
- source identity と content hash による文書バージョン管理
- lexical / metadata / optional vector を組み合わせた Hybrid Search
- PostgreSQL + pgvector による検索基盤
- SQLite を使った軽量なローカル起動
- Evidence / Citation 検証付き Grounded QA
- PDF 原本ページへの bbox ハイライト
- sparse / scan ページ向け Adaptive Vision extraction
- 根拠不足・矛盾時の Conservative Abstention
- API キー未設定でも起動可能な no-cost fallback

## アーキテクチャ

```text
PDF / Markdown
      |
      v
Ingestion -> Document / Version / Page / Chunk -> PostgreSQL + pgvector
                                                   |
自然言語 Query -> lexical + metadata + vector retrieval
                                                   |
                                                   v
                                  Grounding / Citation validation
                                                   |
                                                   v
                                    Answer + Evidence + Source Viewer
```

大きくは次の3段階です。

1. **Ingestion** — PDF / Markdown を version・page・chunk 単位へ分解し、text・bbox・provenance・embedding を保存します。
2. **Hybrid Search** — 図番や階、title などの exact signal と、必要に応じて pgvector の cosine similarity を組み合わせて候補を取得します。
3. **Grounded QA** — 候補 Evidence を LLM へ渡し、返ってきた Evidence ID や数値・関係性をサーバー側でも検証します。十分な根拠がなければ推測せず abstain します。

Gemini 連携は optional です。API キーがない場合でも PDF text layer / Markdown ingestion、lexical / metadata search、保守的なローカル QA fallback は利用できます。`GEMINI_API_KEY` が空のときは Gemini client を作成しません。

## 技術スタック

- Python 3.12+
- FastAPI
- PostgreSQL
- SQLAlchemy / Alembic
- pgvector
- Gemini API（任意）
- PyMuPDF
- SQLite（軽量ローカル実行用）

## セットアップ

### 最小構成：SQLite + 架空サンプル

外部 AI API を使わずに動作確認する場合はこちらが最短です。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m scripts.bootstrap
./start.sh
```

起動後:

- App: http://localhost:8000
- OpenAPI / Swagger UI: http://localhost:8000/docs

### PostgreSQL + pgvector

```bash
docker compose up -d db
cp .env.example .env
python -m alembic upgrade head
python -m scripts.bootstrap
./start.sh
```

`docker-compose.yml` の認証情報はローカル開発用の初期値です。公開環境や本番環境では再利用しないでください。

## Gemini API

Gemini を利用する場合は、各利用者が自分の API キーを設定してください。

`.env.example` を `.env` へコピーし、次を設定します。

```dotenv
GEMINI_API_KEY=
```

このリポジトリには実 API キーを含めていません。また、frontend JavaScript へ API キーを渡す実装や、公開者の API キーを使う hosted endpoint も含みません。

API キー未設定時:

- アプリは起動できます
- Gemini API への request は送信しません
- lexical / metadata search は利用できます
- Markdown / PDF text layer ingestion は利用できます
- Vision / embedding / generated answer は無効化、または保守的なローカル fallback を使います

API キー設定時は Vision extraction・embedding・generated QA などが有効になります。これらは設定した利用者自身の quota / billing を使用するため、モデルや利用量によっては料金が発生する場合があります。

`scripts.bootstrap` は demo setup を no-cost で再現できるよう、shell に API キーが存在していても Vision / embedding を明示的に無効化します。

## デモデータ

`samples/demo-building/` 以下はすべて、この公開リポジトリ用に作成した架空データです。

試せる query 例:

- `受変電設備`
- `杭の工法と支持層`
- `変更資料CHG-001の内容は？`

実在する図面セットからの転載ではありません。

## Grounding 設計

QA で使う各 Evidence にはサーバー側で管理する Evidence ID を付与します。LLM が自由に図番・page・source metadata を作るのではなく、prompt 内で渡された Evidence ID だけを参照させ、その後サーバー側で Citation を再構築します。

主な検証対象:

- unknown Evidence ID
- 数値・単位
- entity / semantic role
- relation
- requested facet の充足
- source provenance

十分に support できない claim は確定せず、必要に応じて abstain します。PDF Evidence は正規化 bbox を保持でき、Citation 選択時に Viewer 上で原本箇所をハイライトします。

公開データを使った評価方針は [docs/EVALUATION.md](docs/EVALUATION.md) を参照してください。

## テスト

live Gemini client を作らず、model behavior は fake client を注入して検証できます。

```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY pytest
```

PostgreSQL integration test は `DATABASE_URL` が利用可能な PostgreSQL / pgvector を指している場合に実行されます。利用できない場合は skip されます。

## 制限事項

- OCR / Vision extraction は図面を誤読する可能性があります
- Grounding guard は hallucination や unsupported claim を減らしますが、完全防止を保証するものではありません
- bbox は canonical には4隅を保持できますが、Viewer では外接矩形として表示します
- production 向け authentication / authorization / multi-tenancy は未実装です
- 同梱 dataset は小規模な synthetic sample です
- table structure の復元は heuristic です

## 今後の改善

- Authentication / Authorization
- Tenant isolation
- Async ingestion / job queue
- Human review workflow
- Table structure extraction
- Observability / tracing / cost control
- Embedding / model version migration
- Backup / disaster recovery

## ライセンス

本プロジェクトの first-party code は [MIT License](LICENSE) で公開しています。

## Third-party notices

第三者由来の UI attribution は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に記載し、source comment も保持しています。

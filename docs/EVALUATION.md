# 公開デモデータを使った評価

評価には `samples/demo-building/` 以下の架空データ、または公開権限を確認できる dataset だけを使用します。

非公開 benchmark の正解データや、公開前の評価結果をこの public repository の評価値として再利用しません。

## Retrieval

小さな query set と relevant document ID を定義し、以下の retrieval 指標を確認します。

- Recall@k
- MRR
- nDCG

期待結果を人間が確認できるよう、query set は公開データと一緒に管理します。

### 見るポイント

- 正解文書が候補へ入っているか
- 最初の正解が十分上位か
- 複数 relevant document の順位が妥当か
- metadata / lexical / vector のどこで取りこぼしているか

## Grounded QA

各質問について、少なくとも次を確認します。

1. 回答が要求された facet に答えているか
2. 重要な claim がすべて Evidence ID に対応しているか
3. Citation の quote が実際の source section / page に存在するか
4. 数値や relation が Evidence に直接 support されているか
5. 根拠がない場合に推測せず abstain できているか

単に文章が自然かではなく、「その回答をその Citation が本当に支えているか」を評価します。

## No-cost test policy

通常の automated test では fake Gemini client を注入し、live model へ request を送りません。

live model を使う evaluation を行う場合は、利用者が明示的に有効化し、自分の API key / quota / budget を使う別プロセスとして扱います。

default test command に live Gemini API call は含めません。

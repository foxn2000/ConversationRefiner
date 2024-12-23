# ConversationRefiner

## 概要

ConversationRefiner は、ユーザーとAIアシスタントの会話データを分析し、より自然で適切な会話へと改善することを目的としたPythonツールです。大規模言語モデル（LLM）を活用し、会話の流れを先読みしたり、ユーザーの意図を深く理解したりすることで、AIアシスタントの応答を洗練させます。

## 機能

*   **会話データの改善:** ユーザーとAIアシスタントの会話履歴を入力として、より自然で、ユーザーの意図に沿った会話データを出力します。
*   **LLMによる推論:**  大規模言語モデルを活用し、文脈を理解した上で適切な改善を行います。
*   **JSONL形式での出力:** 改善された会話データは、JSONL形式でリアルタイムにファイルに保存されます。
*   **複数のLLM APIに対応 (予定):**  `ollama`, `GPT`, `Groq` など、複数のLLM APIを切り替えて利用できます。

## 環境構築

1. **Pythonのインストール:**  Python 3.8 以上がインストールされていることを確認してください。まだインストールされていない場合は、[Python公式サイト](https://www.python.org/downloads/) からダウンロードしてインストールしてください。

2. **仮想環境の作成:**  以下のコマンドで仮想環境を作成し、有効化します。

    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate  # Windows
    ```

3. **依存ライブラリのインストール:**  `requirements.txt` に記述されている依存ライブラリをインストールします。

    ```bash
    pip install -r requirements.txt
    ```

4. **環境変数の設定:**  APIキーなどの環境変数を `.env` ファイルに設定します。リポジトリのルートディレクトリに `.env` ファイルを作成し、以下の内容を記述してください。

    ```.env
    GROQ_API_KEY="your-groq-api-key"
    XAI_API_KEY="your-xai-api-key"
    ```

    **注意:**  `your-groq-api-key` と `your-xai-api-key` は、ご自身のAPIキーに置き換えてください。

## 使い方

1. **`analyze_conversation` 関数の実行:**  改善したい会話データのリストを 'chat_hisstory.jsonl' に保存します。
2. 

    `output_file` 引数で、改善された会話データを保存するJSONLファイルのパスを指定できます。

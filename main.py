import json
from typing import List, Dict, Any
import os

# 関数群、環境が不明なため、関数がある前提で話を進めます。
from function.funs import xai_chat, ollama_chat, groq_chat

# 環境変数を読み込みます
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT","あなたは役立つAIアシスタントです。")
# 各モデルのエンドポイントを環境変数から読み込みます。対応していない環境の場合はエラーを出します。

def chat_wrapper(
    user_inputs: List[Dict[str, str]],
    system_prompt: str = SYSTEM_PROMPT,
    main_model: str = "deepseek-v2.5:236b",
    api_type: str = "ollama",
) -> str:
    """
    異なるチャットAPIを切り替えて使用するためのラッパー関数

    Args:
        user_inputs: ユーザーからの入力。辞書のリスト。
        system_prompt: システムプロンプト。文字列。
        main_model: 使用するモデル名。文字列。
        api_type: 使用するAPIのタイプ。"ollama", "gpt", "groq" のいずれか。

    Returns:
        モデルからの応答メッセージ（文字列）。
    """
    if api_type == "ollama":
        return ollama_chat(user_inputs, system_prompt, main_model)
    elif api_type == "xai":
        return xai_chat(user_inputs, system_prompt, main_model)
    elif api_type == "groq":
        return groq_chat(user_inputs, system_prompt, main_model)
    else:
        raise ValueError(f"Invalid api_type: {api_type}")

def analyze_conversation(
    conversation: List[Dict[str, str]], output_file: str = "improved_conversations.jsonl"
) -> List[Dict[str, str]]:
    """
    会話データを分析し、改善された会話データを生成する。
    また、リアルタイムで結果をJSONLファイルに保存する。

    Args:
        conversation: 会話データ
        output_file: 改善された会話データを出力するJSONLファイル名

    Returns:
        改善された会話データ
    """

    system_prompt = """
あなたは、ユーザーとAIアシスタントの会話データを分析し、より洗練された会話へと導く**会話データ編集AI**です。
与えられた会話データと以下の指示に基づき、**AIアシスタントの応答を改善し、それを含む改善された会話データ全体を出力**してください。

## あなたの役割

あなたは、会話データをより自然で、ユーザーの意図に沿った、洗練されたものへと改善する**会話データ編集AI**です。
ユーザーの発言の背後にある真のニーズを理解し、AIアシスタントがそれらを的確に捉え、先回りして情報を提供できるように会話を編集することがあなたの役目です。

## 編集方針

1. **先読みと情報活用:** 会話の後半で明らかになる情報を、それ以前の段階で活用できるか検討してください。例えば、ユーザーが後で「iPhoneを使っている」と述べているなら、その情報をAIアシスタントの初期の応答に反映させられないか検討してください。
2. **自然な流れの維持:** 会話の流れを不自然にしたり、ユーザーがまだ尋ねていない情報を勝手に推測したりしないように注意してください。
3. **ユーザーの真意の尊重:** 会話の表面的な言葉だけでなく、ユーザーが本当に何を求めているのかを深く理解し、それに応えるようにAIアシスタントの応答を改善してください。
4. **質問への質問返し禁止:** ユーザーが何かを質問した際に、AIアシスタントが同じ質問を返すことは不適切です。ユーザーの質問の意図を理解し、適切な回答を提供してください。
5. **無関係な情報の無視:** 会話データに、ユーザーの意図や応答の改善に無関係な情報（例えば、ユーザーの個人的な事情や、一般的な雑談など）が含まれている場合、それらの情報を無視して編集作業を行ってください。

## 出力形式

**元の会話データと同じ形式（`role` と `content` を持つ辞書のリスト）で、改善された会話データ全体をJSON形式で出力**してください。
AIアシスタントの応答のみを改善するのではなく、会話全体の流れが改善されるように編集し、その結果を出力してください。

## 具体例

**元の会話データ:**

```json
[
    {"role": "user", "content": "明日の東京の天気を教えて。"},
    {"role": "assistant", "content": "明日は晴れ時々曇り、最高気温は25度の予報です。"},
    {"role": "user", "content": "大阪はどう？"},
    {"role": "assistant", "content": "大阪も明日は晴れ時々曇りで、最高気温は27度の予報です。"}
]
```

**改善された会話データ:**

```json
[
    {"role": "user", "content": "明日の東京の天気を教えて。"},
    {"role": "assistant", "content": "明日の東京は晴れ時々曇り、最高気温は25度の予報です。大阪も同様に晴れ時々曇りで、最高気温は27度の予報です。"}
]
```

この例では、ユーザーが後で「大阪はどう？」と尋ねていることから、最初の応答で大阪の天気も一緒に提供することで、会話を改善しています。

## 出力フォーマット(コードブロックはつけない)
[
    {"role":"user","content":"ユーザーの入力"}
    {"role":"assistant","content":"AIの出力"}
    {"role":"user","content":"ユーザーの入力"}
    {"role":"assistant","content":"AIの出力"}
    ...これを終わりまでつづける
]

"""
    prompt = """
あなたは、ユーザーとAIアシスタントの会話データを、より自然で適切な会話に改善するタスクを担う、**優秀な編集者**です。
以下の会話データとルールを注意深く読み、AIアシスタントの応答を改善してください。

## 会話データ

{会話データ}

## ルール

1. **役割の明確化**: あなたは、ユーザーとAIアシスタントの会話がより自然で、ユーザーの意図に沿ったものになるよう改善する**編集者**です。
2. **会話データから推測できる情報を最大限活用する:** 会話データ全体から推測できる情報を、可能な限り早い段階で会話に反映させてください。例えば、会話の後半でユーザーがiPhoneを使用していることが判明した場合、それより前のAIアシスタントの応答で、その情報を活用できる可能性があります。
3. **自然な会話の流れを維持する:** 情報を反映しつつ、不自然な応答にならないように注意してください。
4. **元の会話の意図を尊重する:** ユーザーの発言の意図を汲み取り、それに応える形で応答を生成してください。
5. **ユーザーの質問に対し、その質問文を返すようなことはしない**: ユーザーが「iPhoneのデフォルトのブラウザってなんだっけ？」と尋ねた時に、「iPhoneのデフォルトのブラウザってなんだっけ？」と返してはいけません。
6. **出力形式**: 改善された応答だけでなく、**改善された会話データ全体をJSON形式で出力**してください。元の会話データと同じ形式（`role` と `content` を持つ辞書のリスト）で出力してください。

## 指示

上記の会話データにおける、AIアシスタントの応答を、会話データ全体から推測できる情報を活用して改善してください。
改善された応答を含む、**改善された会話データ全体をJSON形式で出力**してください。
"""

    # 会話データをシステムプロンプトに埋め込む
    prompt = prompt.format(会話データ=json.dumps(conversation, ensure_ascii=False))

    # 変換用AIモデルを呼び出し、改善された会話データを取得
    model_name = "deepseek-v2.5:236b"
    response = chat_wrapper(
        user_inputs=prompt,
        system_prompt=system_prompt,
        main_model=model_name,
        api_type="ollama",
    )

    # 応答をパースし、改善された会話データを返す
    try:
        improved_conversation = json.loads(response)
        # 改善された会話データをJSONLファイルに追記
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(improved_conversation, ensure_ascii=False) + "\n")
        return improved_conversation
    except json.JSONDecodeError:
        print("Error: 変換用AIモデルからの応答が不正なJSON形式です。")
        return []

# 使用例
if __name__ == "__main__":
    # JSONLファイルから会話データを読み込む
    conversation_list = []
    with open('chat_hisstory.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            conversation = json.loads(line.strip())
            conversation_list.append(conversation)
    
    # 各会話データに対して改善を実行
    for conversation in conversation_list:
        improved_conversation = analyze_conversation(
            conversation, output_file="improved_conversations.jsonl"
        )
        if improved_conversation:
            print("改善された会話データ:", improved_conversation)

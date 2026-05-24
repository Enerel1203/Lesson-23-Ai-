import requests
from config import HF_API_KEY

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def analyze_sentiment(text):
    payload = {
        "inputs": text
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload
    )

    print("Status code:", response.status_code)

    try:
        result = response.json()

    except Exception:
        print("Raw response:")
        print(response.text)
        return "Error", "API returned invalid response"

    if "error" in result:
        return "Error", result["error"]

    if isinstance(result, list):
        predictions = result[0]

        best = max(
            predictions,
            key=lambda x: x["score"]
        )

        return (
            best["label"],
            round(best["score"] * 100, 2)
        )

    return "Error", result


while True:
    sentence = input(
        "\nEnter text (or type quit): "
    )

    if sentence.lower() == "quit":
        break

    sentiment, confidence = analyze_sentiment(sentence)

    print("Sentiment:", sentiment)
    print("Confidence:", confidence)
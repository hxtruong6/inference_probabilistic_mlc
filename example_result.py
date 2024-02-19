import json

data = {
    "emotions": {
        "LogisticRegression": {
            "Predict Hamming Loss": {
                "Hamming Loss": ["0.21989", "0.20308", "0.20868", "0.17797", "0.21469"],
                "Subset Accuracy": [
                    "0.19328",
                    "0.31933",
                    "0.26891",
                    "0.31356",
                    "0.26271",
                ],
            },
            "Predict Subset": {
                "Hamming Loss": ["0.21008", "0.22409", "0.21989", "0.19350", "0.21186"],
                "Subset Accuracy": [
                    "0.29412",
                    "0.33613",
                    "0.28571",
                    "0.32203",
                    "0.32203",
                ],
            },
        }
    },
    "CHD_49": {
        "LogisticRegression": {
            "Predict Hamming Loss": {
                "Hamming Loss": ["0.30030", "0.32883", "0.28979", "0.29279", "0.25826"],
                "Subset Accuracy": [
                    "0.19820",
                    "0.14414",
                    "0.17117",
                    "0.18919",
                    "0.25225",
                ],
            },
            "Predict Subset": {
                "Hamming Loss": ["0.31381", "0.32282", "0.28979", "0.29580", "0.27477"],
                "Subset Accuracy": [
                    "0.18919",
                    "0.17117",
                    "0.19820",
                    "0.19820",
                    "0.23423",
                ],
            },
        }
    },
}

# Convert the data to a JSON object
json_data = json.dumps(data, indent=4)  # Use indent for pretty-printing

# Print the JSON object
print(json_data)

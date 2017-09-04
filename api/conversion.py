[{'faceRectangle': {'height': 367, 'left': 39, 'top': 48, 'width': 367}, 'scores': {'anger': 0.0005124948, 'contempt': 0.0409049541, 'disgust': 0.00170815631, 'fear': 0.000130237953, 'happiness': 0.03902028, 'neutral': 0.880412638, 'sadness': 0.00764320651, 'surprise': 0.0296680536}}]

TRANSLATE = {
    'anger': '憤怒',
    'contempt': '蔑視',
    'disgust': '討厭',
    'fear': '恐懼',
    'happiness': '快樂',
    'neutral': '中立',
    'sadness': '憂傷',
    'surprise': '驚奇'
}

a = [
    {
        "faceRectangle": {
            "height": 162,
            "left": 177,
            "top": 131,
            "width": 162
        },
        "scores": {
            "anger": 5.198238e-05,
            "contempt": 0.000133138747,
            "disgust": 1.53003639e-05,
            "fear": 6.312813e-05,
            "happiness": 0.000270753721,
            "neutral": 0.988568544,
            "sadness": 0.007187182,
            "surprise": 0.0037099754
        }
    }
]

scores = a[0]['scores']
total = 0

for score in scores:
    scoreValue = format(scores[score], 'f')

    print('{}: {}'.format(TRANSLATE[score], scoreValue))



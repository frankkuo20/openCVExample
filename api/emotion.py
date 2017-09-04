import json
import requests

API_KEY = '67a3bbe8bb604a2cb30cc7f198b2dfac'
URL_BASE = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'

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


class Recognize:
    def __init__(self, imagePath):
        if imagePath[:4] == 'http':
            self._requestParams = self._getRequestParamsUrl(imagePath)
        else:
            self._requestParams = self._getRequestParamsLocal(imagePath)

    def _getRequestParamsUrl(self, imagePath):
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': API_KEY,
        }
        body = {
            'url': imagePath
        }
        requestParams = {
            'headers': headers,
            'json': body,
            'params': {}
        }
        return requestParams

    def _getRequestParamsLocal(self, imagePath):
        f = open(imagePath, 'rb')
        body = f.read()
        f.close()

        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': API_KEY,
        }

        requestParams = {
            'headers': headers,
            'data': body,
            'params': {}
        }
        return requestParams

    def run(self):
        try:
            # Execute the REST API call and get the response.
            response = requests.request('POST', URL_BASE, **self._requestParams)

            print('Response:')
            parsed = json.loads(response.text)
            data = json.dumps(parsed, sort_keys=True, indent=2)
            print(data)
            self.printTran(parsed)

        except Exception as e:
            print('Error:')
            print(e)

    def printTran(self, data):
        '''

        :param data: json format
        :return:
        '''

        for index, people in enumerate(data):
            scores = people['scores']
            print('People: {}'.format(index + 1))
            maxScore = 0
            maxScoreText = ''
            for score in scores:
                scoreValue = scores[score]
                score = TRANSLATE[score]
                if scoreValue > maxScore:
                    maxScore = scoreValue
                    maxScoreText = score

                scoreValue = format(scoreValue, 'f')
                print('{}: {}'.format(score, scoreValue))

            print('Result: {}: {}'.format(maxScoreText, maxScore))


if __name__ == '__main__':
    imagePath = 'https://upload.wikimedia.org/wikipedia/commons/c/c3/RH_Louise_Lillian_Gish.jpg'
    imagePath = './image/User.1.1.jpg'

    recognize = Recognize(imagePath)
    recognize.run()
    # data = [{"faceRectangle": {
    #     "height": 162, "left": 177, "top": 131, "width": 162},
    #     "scores": {
    #         "anger": 5.198238e-05,
    #         "contempt": 0.000133138747,
    #         "disgust": 1.53003639e-05,
    #         "fear": 6.312813e-05,
    #         "happiness": 0.000270753721,
    #         "neutral": 0.988568544,
    #         "sadness": 0.007187182,
    #         "surprise": 0.0037099754
    #     }
    # }]
    # recognize.printTran(data)

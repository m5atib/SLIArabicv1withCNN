import cv2
from tensorflow import keras
import numpy as np
from cvzone.HandTrackingModule import HandDetector

model = keras.models.load_model('model.h5')

letters = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf',
           'ghayn', 'ha', 'haa', 'jim', 'kaf', 'kha', 'laa', 'laam', 'mim',
           'nun', 'ra', 'saad', 'seen', 'sheen', 'taa', 'taaa', 'thaa', 'thal',
           'toot', 'waw', 'ya', 'yaa', 'zay']

images = []
for c in letters:
    images.append(cv2.imread('arabicletters/' + c + '.png'))

x, y, w, h = 0, 0, 0, 0

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
indexoflabel = 0
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #hands = detector.findHands(img,draw=False)  # with draw
    hands, img = detector.findHands(img)  # without draw

    if hands:
        x, y, w, h = hands[0]["bbox"]  # Bounding box info x,y,w,h
        crop_img = img[y - 15:y + h + 15, x - 15:x + w + 15]
        try:
        # print("YES")
            frame = cv2.resize(crop_img, (64, 64))
            frame = frame.reshape( [-1, 64, 64, 3])
            frame = frame / 255.0
            predictions = model.predict(frame)
            indexoflabel = np.argmax(predictions)
            cv2.imshow("Label", images[indexoflabel])
        except: pass
    img = cv2.putText(img, "Letter : " + letters[indexoflabel],
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #img = cv2.rectangle(img, (x - 35, y - 35), (x + w + 35, y + h + 35), (0, 255, 0), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

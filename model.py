from sklearn.svm import SVC
import numpy as np
import cv2 as cv
import PIL


def preprocess_image(img):
    # Preprocess the image:
    # 1. Convert to grayscale
    # 2. Resize to a consistent size
    # 3. Flatten the image into a 1D array
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)  # Adjust kernel size as needed
    img = cv.resize(img, (150, 150))
    img = img.flatten()
    return img


class Model:
    def __init__(self, num_classes):
        self.model = SVC(kernel='rbf', class_weight='balanced', C=1.0)
        self.num_classes = num_classes

    def train_model(self, class_counters):
        img_list = []
        class_list = []

        for class_num, counter in enumerate(class_counters):
            for i in range(0, counter + 1):
                img = cv.imread(f'{class_num}/frame{i + 1}.jpg')
                img = preprocess_image(img)
                img_list.append(img)
                class_list.append(class_num)

        if len(img_list) == 0:
            print("No training data found.")
            return

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", frame)
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')
        img = preprocess_image(img)  # Use preprocess_image method
        prediction = self.model.predict([img])

        return prediction[0]

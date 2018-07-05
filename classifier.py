import numpy as np
from PIL import  Image
import os, cv2

# Method to train custom classifier to recognize face
def train_classifer(data_dir):
    # Read all the images in custom data-set
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")


train_classifer("data")
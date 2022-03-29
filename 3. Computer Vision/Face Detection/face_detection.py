# This is going to be a "Face detection" model, not a "Face recognition/verification" model. Train a primitive
# model (scaled down version of ResNet-34) from scratch using some 10,000 images of faces from a dataset? And the
# goal is not to name the faces, but to draw a bounding box around them if found.
#
# There are primarily two ways to do this: 1. Use a pretrained Cascade Classifier (e.g. Haar Cascade, LBP,
# etc.). This is the classical approach without any deep learning. Or sometimes, Histogram of Oriented Gradients (
# HOG) features. 2. Use a deep learning based model (e.g. ResNet-34, Inception, etc.). This is the modern approach.

from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt

image_path = 'kim_tae_ri.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = MTCNN()
faces = detector.detect_faces(img)
print(faces)

img_with_faces = img.copy()
min_conf = 0.9
for det in faces:
    if det['confidence'] >= min_conf:
        x, y, width, height = det['box']
        keypoints = det['keypoints']
        cv2.rectangle(img_with_faces, (x, y), (x + width, y + height), (0, 155, 255), 2)
        cv2.circle(img_with_faces, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(img_with_faces, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(img_with_faces, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(img_with_faces, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(img_with_faces, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
# plt.figure(figsize=(10, 10))
plt.imshow(img_with_faces)
plt.axis('off')
plt.show()

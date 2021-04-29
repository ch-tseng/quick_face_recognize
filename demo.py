import time
import cv2
import imutils
from libFaceRecog import FACE_RECOG

remake_db = True
facenet_model_path = 'models/facenet_keras_weights.h5'
embedding_db_path = 'embeddings/'
faces_images_db_source = 'Faces/'
recog_threshold = 0.5

FACE = FACE_RECOG(facenet_path=facenet_model_path)

if remake_db is True:
    FACE.make_db( faces_path=faces_images_db_source, db_path=embedding_db_path )

FACE.load_db(embedding_db_path)

start = time.time()
face_names = FACE.recognize_knn(embedding_db_path, cv2.imread('test.jpg'))
for data in face_names:
    print(data)

print('Used time:', time.time()-start)
print('-------------------------------------------------------------------------------')


start = time.time()
face_names = FACE.recognize(cv2.imread('test.jpg'),recog_threshold)

#for [name, fimg, fbox] in face_names:
#    print(name, fbox)
#    cv2.imshow('test', fimg)
#    cv2.waitKey(0)

print('Used time:', time.time()-start)


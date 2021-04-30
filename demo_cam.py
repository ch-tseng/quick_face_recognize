import time
import cv2
import imutils
from libFaceRecog import FACE_RECOG
from PIL import ImageFont, ImageDraw, Image
import numpy as np

remake_db = True
facenet_model_path = 'models/facenet_keras_weights.h5'
embedding_db_path = 'embeddings/'
faces_images_db_source = 'Faces/'
recog_threshold = 0.5
onlyone = True

def printText(bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "fonts/wt009.ttf"
        font = ImageFont.truetype(fontpath, int(size*10*4))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg


if __name__ == "__main__":

    FACE = FACE_RECOG(facenet_path=facenet_model_path)

    if remake_db is True:
        FACE.make_db( faces_path=faces_images_db_source, db_path=embedding_db_path )

    FACE.load_db(embedding_db_path)


    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,frame = cap.read()

        #face_names = FACE.recognize_knn(embedding_db_path, frame, threshold=recog_threshold, onlyone=onlyone)
        face_names = FACE.recognize(frame,recog_threshold)

        for data in face_names:
            print('data', data)
            fname = data[0]
            diff = data[1]
            bbox = data[2]

        fname = ''
        if onlyone is True:
            if len(FACE.recog_name_list)>0:
                fname = max(FACE.recog_name_list ,key=FACE.recog_name_list.count)

        if len(fname)>0:
            frame = printText(frame, fname + ':' + str(round(diff,3)), color=(0,255,0,0), size=1.25, pos=(bbox[0],bbox[1]-30), type="Chinese")


        cv2.imshow('test', frame)
        cv2.waitKey(1)


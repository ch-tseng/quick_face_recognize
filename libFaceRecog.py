import os
import cv2
import math
import time
import imutils
import numpy as np
import pickle 
from mtcnn_cv2 import MTCNN
from sklearn.preprocessing import Normalizer
from architecture import *
from scipy.spatial.distance import cosine
from tqdm import tqdm
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier

class FACE_RECOG:
    def __init__(self, facenet_path):
        self.detector = MTCNN()
        self.required_shape = (160,160)
        self.face_encoder = InceptionResNetV2()
        self.face_encoder.load_weights(facenet_path)
        self.l2_normalizer = Normalizer('l2')

        self.last_recog_time = 0.0
        self.recog_group_max = 5
        self.recog_name_list = []
        self.clear_recog_max_time = 3  #seconds

    def normalize(self, img):
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def good_gesture(self, keypoints):
        if len(keypoints) != 5:
            return False

        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        nose = keypoints['nose']
        mouth_left = keypoints['mouth_left']
        mouth_right = keypoints['mouth_right']

        center_eye_x = int((left_eye[0]+right_eye[0])/2)
        x_range = abs(right_eye[0]-left_eye[0]) / 5
        center_x_range = (center_eye_x-x_range , center_eye_x+x_range)

        if nose[0]<center_x_range[1] and nose[0]>center_x_range[0]:
            return True
        else:
            return False

    def get_angle(self, a, b, c):
        ang = round(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])), 2)
        return ang + 360 if ang < 0 else ang

    def align_face(self, img_face, eyepoints):
        #cv2.imshow('test', img_face)
        left_eye = eyepoints[0]
        right_eye = eyepoints[1]
        angle = self.get_angle(right_eye, left_eye, (left_eye[1], 99999))
        #print('angle', angle)
        img = imutils.rotate(img_face,90-angle)
        #cv2.imshow('test2', img)

        #cv2.waitKey(0)
        return img

    def get_faces(self, img):
        face_detector = self.detector
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_detector.detect_faces(img_RGB)

        bboxes, fimages = [], []
        for face_info in result:
            [fx,fy,fw,fh] = face_info['box']
            if fw<self.required_shape[0] and fh<self.required_shape[1]:
                continue

            #bboxes.append(face_info['box'])
            keypoints = face_info['keypoints']
            if self.good_gesture(keypoints) is True:
                pad = int(fh/4)
                fx -= pad
                fy -= pad
                fw += (pad*2)
                fh += (pad*2)
                if fx<0: fx=0
                if fy<0: fy=0
                if fw>img.shape[1]: fw=img.shape[1]
                if fh>img.shape[0]: fh=img.shape[0]
                face_area = img[fy:fy+fh, fx:fx+fw]

                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']
                aligned_img = self.align_face(face_area, (left_eye,right_eye))
                #cv2.imshow('test', aligned_img)
                #cv2.waitKey(0)

                img_RGB = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
                aligned_result = face_detector.detect_faces(img_RGB)
                if len(aligned_result)>0:
                    aligned_bbox = aligned_result[0]['box']
                    ax,ay,aw,ah = aligned_bbox[0], aligned_bbox[1], aligned_bbox[2], aligned_bbox[3]
                    #bboxes.append(aligned_bbox)
                    bboxes.append(face_info['box'])
                    fimages.append( aligned_img[ay:ay+ah, ax:ax+aw] )

        return zip(bboxes, fimages)

    def get_embeddings(self, faceimg):
        face_encoder = self.face_encoder
        img_RGB = cv2.cvtColor(faceimg, cv2.COLOR_BGR2RGB)
        face = cv2.resize(faceimg, self.required_shape)
        face = self.normalize(face)
        face_d = np.expand_dims(face, axis=0)

        encode = face_encoder.predict(face_d)[0]

        return encode

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            encoding_dict = pickle.load(f)
        return encoding_dict

    def load_db(self, db_path):
        self.embedding_dict = self.load_pickle( os.path.join(db_path,'encodings.pkl'))

    def embedding_compare(self, embed, name_check):
        distance = float("inf")
        min_diff = 99999
        for db_name, db_encode in self.embedding_dict.items():
            if db_name == name_check:
                dist = cosine(db_encode, embed)
                print(db_name, dist)
                if dist<min_diff:
                    min_diff = dist

        return min_diff

    def recognize_knn(self, db_path, img, onlyone=True):
        clf = load(os.path.join(db_path, 'knn_faces.joblib'))
        face_encoder = self.face_encoder
        faces_data = self.get_faces(img=img)
        #if onlyone is True and len(faces_data)>1:
        #    return []

        boxes = []
        preds = []
        diffs = []
        if(time.time() - self.last_recog_time > self.clear_recog_max_time):
            self.recog_name_list = []

        for id, [fbox, fimg] in enumerate(faces_data):
            if (onlyone is True and id==0) or (onlyone is False):
                encode = self.get_embeddings(fimg)
                pred = clf.predict( np.array([encode]) )
                preds.append(pred[0])
                diffs.append(self.embedding_compare(encode, pred[0]))
                boxes.append(fbox)

                self.last_recog_time = time.time()
                self.recog_name_list.append(pred[0])
                if len(self.recog_name_list)>self.recog_group_max: self.recog_name_list.pop(0)

                print('test', self.recog_name_list)

        return zip(preds, diffs, boxes)

    def recognize(self, img, threshold, onlyone=True):
        face_encoder = self.face_encoder
        faces_data = self.get_faces(img=img)

        names, face_imgs, face_boxes = [], [], []
        if(time.time() - self.last_recog_time > self.clear_recog_max_time):
            self.recog_name_list = []
        for [fbox, fimg] in faces_data:
            if (onlyone is True and id==0) or (onlyone is False):
                encode = self.get_embeddings(fimg)
                encode = self.l2_normalizer.transform(encode.reshape(1, -1))[0]
                name = 'unknown'

                distance = float("inf")
                #print('test', len(self.embedding_dict))
                for db_name, db_encode in self.embedding_dict.items():
                    dist = cosine(db_encode, encode)
                    print(db_name, dist)
                    if dist < threshold and dist < distance:
                        name = db_name
                        distance = dist

                names.append(name)
                face_imgs.append(fimg)
                face_boxes.append(fbox)

                self.last_recog_time = time.time()
                self.recog_name_list.append(pred[0])
                if len(self.recog_name_list)>self.recog_group_max: self.recog_name_list.pop(0)

        return zip(names, face_imgs, face_boxes)


    def make_db(self, faces_path, db_path):
        face_encoder = self.face_encoder

        if not os.path.exists(db_path):
            os.makedirs(db_path)

        encoding_dict = dict()
        encodes = []
        names = []
        knnencodes = []
        for face_names in tqdm(os.listdir(faces_path)):
            person_dir = os.path.join(faces_path,face_names)

            id = 0
            for image_name in os.listdir(person_dir):
                if image_name[-3:] not in ['jpg', 'png', 'peg']:
                    continue

                id += 1
                image_path = os.path.join(person_dir,image_name)

                img_BGR = cv2.imread(image_path)
                faces_data = self.get_faces(img=img_BGR)

                #[box, fimg] = faces_data[0]
                for fid, [box, fimg] in enumerate(faces_data):
                    if fid == 0:
                        cv2.imwrite(os.path.join(db_path, face_names+'_'+str(id)+'.jpg'), fimg)
                        encode = self.get_embeddings(faceimg=fimg)
                        encodes.append(encode)
                        knnencodes.append(encode)
                        names.append(face_names)

            #print('test2', len(encodes))
            if encodes:
                encode = np.sum(encodes, axis=0 )
                encode = self.l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                encoding_dict[face_names] = encode

        path = os.path.join(db_path,'encodings.pkl')
        with open(path, 'wb') as file:
            pickle.dump(encoding_dict, file)


        #KNN
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(np.array(knnencodes), np.array(names))
        dump(knn, os.path.join(db_path, 'knn_faces.joblib'))

import os
import cv2
import numpy as np
import torch
from sklearn.svm import SVC
from facenet_pytorch import InceptionResnetV1
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    known_face_encodings = [] 
    known_face_labels = []    

    known_faces_dir = 'dataset'

    for filename in os.listdir(known_faces_dir):
        image_path = os.path.join(known_faces_dir, filename)
        label = os.path.splitext(filename)[0]

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_tensor = torch.tensor(image_rgb.transpose((2, 0, 1)), dtype=torch.float32)

        face_tensor = face_tensor.unsqueeze(0)

        face_embedding = resnet(face_tensor).detach().numpy()

        known_face_encodings.append(face_embedding.flatten())
        known_face_labels.append(label)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.6,1.1],
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    model = SVC(kernel='linear')
    model.fit(known_face_encodings, known_face_labels)

    prototxt_path = "opencv_face_detector.pbtxt"
    model_path = "opencv_face_detector_uint8.pb"
    net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)

                face_region = frame[startY:endY, startX:endX]
                try:
                    face_region_preprocessed = cv2.resize(face_region, (160, 160))
                except: 
                    break
                
                face_region_rgb = cv2.cvtColor(face_region_preprocessed,cv2.COLOR_BGR2RGB)
                face_tensor = torch.tensor(face_region_rgb.transpose((2, 0, 1)), dtype=torch.float32)

                face_tensor = face_tensor.unsqueeze(0)

                face_embedding = resnet(face_tensor).detach().numpy()

                predicted_label = model.predict([face_embedding.flatten()])[0]
                cv2.putText(frame, predicted_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

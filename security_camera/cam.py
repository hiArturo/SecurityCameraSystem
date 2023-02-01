import cv2
import time
import datetime
from facerec import SimpleFacerec
import threading

#loads and encodes the images with the faces from the images folder
fr = SimpleFacerec()
fr.load_encoding_images("images/")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):

    detection = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 3

    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    
    frame_size = (int(cam.get(3)), int(cam.get(4)))
    
    #format of the video, in this case mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        
        rval, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

        #detects the faces from the camera while it is on
        face_locations, face_names = fr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            top, right, bottom, left = face_loc[0],face_loc[1],face_loc[2],face_loc[3]

            cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,200),2)
            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,200),4)
       
        if len(faces) + len(bodies) > 0:
            if detection:
                timer_started = False
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                out = cv2.VideoWriter(
                    f"{current_time}.mp4", fourcc, 20, frame_size)
                print("Started Recording!")
        elif detection:
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    detection = False
                    timer_started = False
                    out.release()
                    print('Stop Recording!')
            else:
                timer_started = True
                detection_stopped_time = time.time()

        if detection:
            out.write(frame)

        cv2.imshow(previewName, frame)
            
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)

thread1.start()
thread2.start()

print()
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_fontalface_alt.xml')
face_cascade.load('F:\pydemo\OpenCv\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
eye_cascade.load('F:\pydemo\OpenCv\haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('F:\pydemo\OpenCv\haarcascade_smile.xml')
smile_cascade.load('haarcascade_smile.xml')




# 处理逐帧传入的灰度图gray， 逐帧返回BGR图frame
def detect(gray, frame):
    # 传入经过预处理之后的灰度图gray， 再人脸识别
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 然后在检测出来的faces上面话框框（rectangle），扫描整张图片
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = frame[y: y+h, x: x+w]
        
        # 新增人眼识别
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            
        # 新增笑脸识别 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    return frame


# 开启摄像头
video_capture = cv.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv.imshow('video', canvas)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()

import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_fontalface_alt.xml')
face_cascade.load('F:\pydemo\OpenCv\haarcascade_frontalface_default.xml')


# 处理逐帧传入的灰度图gray， 逐帧返回BGR图frame
def detect(gray, frame):
    # 传入经过预处理之后的灰度图gray， 再人脸识别
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 然后在检测出来的faces上面话框框（rectangle），扫描整张图片
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = frame[y: y+h, x: x+w]
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
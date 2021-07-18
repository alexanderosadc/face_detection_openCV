import cv2 as cv
# Set source of image camera
imgCapture = cv.VideoCapture(0)

# Set width and height of image 640x480
imgCapture.set(3, 640)
imgCapture.set(4, 480)

# Set haar clasifier. Haar cascades using the edges tries to determine if it is the face.
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Read each frame. Each image return 2 values, if the frame was captured and the frame itself
    isTrue, frame = imgCapture.read()
    # Making transform in gray color. The face detection does not involve colors so we put monotone.
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply face detection function. Scale factor is responsible for scaling the rectangular when we move fwd or bck.
    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=6)

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show image
    cv.imshow('Video', frame)

    # When we press d the loop is stoping
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

img.release()
cv.destroyAllWindows()

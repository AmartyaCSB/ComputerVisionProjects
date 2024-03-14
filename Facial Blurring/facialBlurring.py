import cv2
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#open webcam
cap = cv2.VideoCapture(0)

# For Video run Infinite loop(FPS)
while True:
    ret, photo = cap.read()
    # For image to be changed to gray scale
    gray = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    face  = model.detectMultiScale(gray, 1.3, 5)
    if len(face) == 0:
        pass
    else:
        # x1, y1, x2 ,y2 coordinates of model (rectangle)
        x1 = face[0][0]
        y1 = face[0][1]
        x2 = face[0][2] + x1
        y2 = face[0][3] + y1 
        # 4 Co-ordinates around the Face i.e x1, y1, x2 ,y2
        coord_img = photo[y1:y2 , x1:x2]
        #Function to Blur of those coordinate points
        blurImg = cv2.blur(coord_img, (50,50))
        # make the blur to apply on image
        photo[y1:y2 , x1:x2] = blurImg
        cv2.imshow("Video Window",photo)
        # Key press ENTER to exit the loop
        if cv2.waitKey(100) == 13:
            break
             
cv2.destroyAllWindows()
# Release the camera 
cap.release()


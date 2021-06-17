import cv2
import numpy as np

#  Retrieves video feed from location within the paranthesis.
# cap = cv2.VideoCapture(r'http://192.168.43.1:8080/video')
cap = cv2.VideoCapture(0)

_, frame1 = cap.read()
_, frame2 = cap.read()

#  Runs as long as the video feed is active.
while cap.isOpened():
    #  diff calculates the difference between the first and second frame. The first acts as an image.
    #  gray converts the image to a gray scale image.
    #  blur applies a gaussian blur that reduces high frequency components/colors.
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
   
   #  dilated basically looks for the are of the picture where darker areas are and centers on that.
   #  contours returns all the points that join together.
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        #  Sets the xand y axes and the width and height.
        (x, y, w, h) = cv2.boundingRect(contour)

        #  Checks if the contours total area is less than 700.
        if cv2.contourArea(contour) < 700:
            continue

        #  Draws rectangle around items of interest.
        #  Adds text to notify the user there are changes in the image/video feed.
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame1, "[*] Status: {}".format('Motion Detected'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0),2)
        
    #  Displays the current video.
    cv2.imshow("Detector", frame1)
    frame1 = frame2
    _, frame2 = cap.read()


    #  If the key "q" is entered the while loop ends.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Teminates the window displaying the video and end the video streaming process.
cv2.destroyAllWindows()
cap.release()
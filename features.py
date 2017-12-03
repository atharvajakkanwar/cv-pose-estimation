import pose
import numpy as np
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('pattern.jpg',0)          # queryImage
img2 = cv2.imread('1.jpg',0) # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()

camera = cv2.VideoCapture()
camera.open(0)
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)

wanted = []
gray = None

while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:3], flags=2, outImg=None)
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches[:4]]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches[:4]]
    for item in list_kp2:
        cv2.circle(gray, (int(item[0]), int(item[1])), 3, (0, 255, 0), -1)
    cv2.imshow("Hello", cv2.drawMatches(img1,kp1,gray,kp2,matches[:3], flags=2, outImg=None))
    wanted.append(list_kp2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.main(wanted,gray)

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()





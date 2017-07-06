#https://stackoverflow.com/questions/39752235/python-how-to-detect-vertical-and-horizontal-lines-in-an-image-with-houghlines-w
import numpy as np
import cv2

gray = cv2.imread('linija.png')
edges = cv2.Canny(gray,50,150,apertureSize = 3)
#cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=10)

a,b,c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    #cv2.imwrite('houghlines5.jpg',gray)

line = [(lines[0][0][0], lines[0][0][1]), (lines[0][0][2], lines[0][0][3])]

print 'Koordinate linije: ' + str(line)
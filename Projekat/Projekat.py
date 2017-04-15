# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import ndimage
from vector import distance, pnt2line2
from matplotlib.pyplot import cm
import itertools
import time

from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('th')

# load json and create model - UCITAVAM MODEL KAKO NE BI NA POCETKU TRENIRAO NN
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

cap = cv2.VideoCapture("videos/video-3.avi")
flag, img = cap.read()

succes, frame = cap.read()
if succes:
    cv2.imwrite('linija.png', frame)

#hough transformacija za pronalazanje linije
gray = cv2.imread('linija.png')
edges = cv2.Canny(gray,50,150,apertureSize = 3)
#cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=200
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180,
                        threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=20)

a,b,c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

line = [(lines[0][0][0], lines[0][0][1]), (lines[0][0][2], lines[0][0][3])]

print 'Koordinate linije: ' + str(line)

#line = [(48,398), (447,98)] #rucno pronadjeno koje su koordinate linije, napisi funkciju

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

#kako povezati objekte izmedju frejmova
def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

# color filter
kernel = np.ones((2,2),np.uint8)
#lower = np.array([230, 230, 230])
#upper = np.array([255, 255, 255])

boundaries = [
    ([230, 230, 230], [255, 255, 255])
]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('videos/video_rez.avi',fourcc, 20.0, (640,480))

elements = []
t =0
counter = 0
times = []
suma = 0
answer = '' #

while (1):
    start_time = time.time()
    ret, img = cap.read()

    frame = 'frames/frame-' + str(t) + '.png'  #snimanje frejmova
    cv2.imwrite(frame, img)
    if not (ret):
        break

    (lower, upper) = boundaries[0]
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    img0 = 1.0 * mask

    img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0, kernel)

    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)

    for i in range(nr_objects):
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                    (loc[0].stop + loc[0].start) / 2)
        (dxc, dyc) = ((loc[1].stop - loc[1].start),
                      (loc[0].stop - loc[0].start))

        if (dxc > 11 or dyc > 11):
            cv2.circle(img, (xc, yc), 16, (25, 25, 255), 1)
            elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
            # find in range
            lst = inRange(20, elem, elements)
            nn = len(lst)
            if nn == 0:
                elem['id'] = nextId()
                elem['t'] = t
                elem['pass'] = False
                elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]
                elem['future'] = []
                elements.append(elem)
            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})
                lst[0]['future'] = []

    for el in elements:
        tt = t - el['t']
        if (tt < 3):
            dist, pnt, r = pnt2line2(el['center'], line[0], line[1])
            if r > 0:
                cv2.line(img, pnt, el['center'], (0, 255, 25), 1)   #zelena linija koja predvidja koji ce elementi preci
                c = (25, 25, 255)   #boja kruga oko elementa
                if (dist < 9):
                    c = (0, 255, 160)   #boja kad predje preko linije
                    if el['pass'] == False:
                        el['pass'] = True
                        counter += 1

                        img = cv2.imread('frames/frame-{}.png'.format(el['history'][0]['t']))   #citam frejmove

                        blok_size = (28, 28)
                        blok_center = (int(el['history'][0]['center'][0]),
                                       int(el['history'][0]['center'][1]))  # centar bloka = centar broja
                        # print (blok_center)
                        blok_loc = (blok_center[1] - blok_size[0] / 2, blok_center[0] - blok_size[1] / 2)
                        imgB = img[blok_loc[0]:blok_loc[0] + blok_size[0], blok_loc[1]:blok_loc[1] + blok_size[1], 0]

                        cv2.imwrite('images/' + str(el['id']) + '.png', imgB)

                        imgB = imgB.reshape(1, 1, 28, 28).astype('float32')

                        # normalize inputs from 0-255 to 0-1
                        imgB = imgB / 255

                        imgB_test = imgB.reshape(784)
                        # print imgB_test
                        imgB_test = imgB_test / 255.
                        # print imgB_test.shape
                        tt = model.predict(imgB, verbose=1)
                        #print tt
                        result = 0
                        answer = np.argmax(tt)
                        #print answer

                        if (el['pass']):
                            suma += answer


            cv2.circle(img, el['center'], 16, c, 2)

            id = el['id']
            cv2.putText(img, str(el['id']),
                        (el['center'][0] + 10, el['center'][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255)   #stampa id kraj broja *plavi broj
            for hist in el['history']:
                ttt = t - hist['t']
                if (ttt < 100):
                    cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

            for fu in el['future']:
                ttt = fu[0] - t
                if (ttt < 100):
                    cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

    elapsed_time = time.time() - start_time
    times.append(elapsed_time * 1000)
    cv2.putText(img, 'Proslo: ' + str(counter), (460, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 50), 2)
    cv2.putText(img, 'Broj: ' + str(answer), (460, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, 'Suma: ' + str(suma), (460, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # print nr_objects
    t += 1
    #if t % 10 == 0: #zak
        #print t
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    out.write(img)

out.release()
cap.release()
cv2.destroyAllWindows()

et = np.array(times)
print 'mean %.2f ms' % (np.mean(et))
# print np.std(et)

print ('Suma = ' + str(suma))


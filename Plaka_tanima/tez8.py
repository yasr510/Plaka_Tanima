import cv2
import numpy as np
import os

kernel = np.ones((4,4),np.uint8)
kernel_a = np.ones((3,3),np.uint8)

#url = ""
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
while (True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    input_image = cv2.medianBlur(input_image,ksize=5)

    th, im_th = cv2.threshold(input_image, 100, 255, cv2.THRESH_BINARY_INV)

    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255); #
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv


    thresh = cv2.morphologyEx(im_out,cv2.MORPH_CLOSE,kernel_a,iterations=1)

    cv2.imshow('tresh', thresh)
    cv2.waitKey(100)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sayı = 0
    for i in contours:
        sayı = sayı +1
        print("sayı: ",sayı)

        rect = cv2.minAreaRect(i)
        kutu = cv2.boxPoints(rect)
        kutu = np.int64(kutu)

        minx = np.min(kutu[:, 0])
        miny = np.min(kutu[:, 1])
        maxx = np.max(kutu[:, 0])
        maxy = np.max(kutu[:, 1])

        copy = thresh[miny:maxy, minx:maxx].copy()
        #copy = cv2.resize(copy,(500,300))
        #cv2.imshow("copy",copy)
        #cv2.waitKey()

        toplam_pixeler = copy.shape[0] * copy.shape[1]
        siyah_pixeler = cv2.countNonZero(copy)
        beyaz_pixeler = toplam_pixeler - siyah_pixeler

        if (toplam_pixeler > 0):

            oran = beyaz_pixeler / toplam_pixeler
            print(f"Siyah pixel oranı: {oran:.2f}")
        if(toplam_pixeler <= 0):
            oran = 1
            print(f"Siyah pixel oranı: {oran:.2f}")
        print(sayı)
        W = maxx - minx
        H = maxy - miny
        oran2 = W/H
        oran3 = H*W
        oran4 = H/W
        print("kontur oranı:",oran2)
        print("alan: ",oran3 )

        if (6000 > oran3 > 1500):
            if (oran < 0.5) and (oran2 > 2.6) and (oran4 > 0.30):
                a = cv2.drawContours(frame, [kutu], 0, (0, 255, 0), 2)
                print("plaka bulundu")

                roi = frame[miny:maxy, minx:maxx]
                cv2.imshow('ROI', roi)
                cv2.waitKey(100)


            else:
                a = cv2.drawContours(frame, [kutu], 0, (0, 0, 255), 2)
                print("plaka bulunamadı")




cap.release()
cv2.destroyAllWindows()
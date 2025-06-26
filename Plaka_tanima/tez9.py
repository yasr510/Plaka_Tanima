import cv2
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"  # Tesseract OCR dizinini belirtin



def perpektif(image,input_array):
    height, width = image.shape[:2]
    input_array = np.float32(input_array)
    output_array = np.float32([(0,0),(width,0),(width,height),(0,width)])
    matrix = cv2.getPerspectiveTransform(input_array,output_array)
    result = cv2.warpPerspective(image,matrix,(width,height),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return result


dosya_yolu = ""
kernel = np.ones((4,4),np.uint8)
im_in = cv2.imread(dosya_yolu)
im_in = cv2.resize(im_in,(700,500))
input_image = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
kernel_a = np.ones((3,3),np.uint8)

input_image = cv2.medianBlur(input_image,ksize=5)
cv2.imshow("original", im_in)
cv2.imshow("blur", input_image)
cv2.waitKey()

im_th = cv2.bilateralFilter(input_image, 11, 17, 17)
cv2.imshow("tresh", im_th)
cv2.waitKey()
im_th = cv2.Canny(im_th, 100, 255)
cv2.imshow("tresh", im_th)
cv2.waitKey()
#th, im_th = cv2.threshold(input_image, 100, 255, cv2.THRESH_BINARY_INV)


im_floodfill = im_th.copy()

h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

cv2.floodFill(im_floodfill, mask, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = im_th | im_floodfill_inv


cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.waitKey()

thresh = cv2.morphologyEx(im_floodfill_inv,cv2.MORPH_CLOSE,kernel_a,iterations=1)
cv2.imshow("closeing",thresh)
cv2.waitKey()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
sayı = 0
for i in contours:
    sayı = sayı + 1
    if ( sayı < 10):
        print(sayı)
        rect = cv2.minAreaRect(i)
        kutu = cv2.boxPoints(rect)
        kutu = np.int64(kutu)

        # minx = np.min(kutu[:, 0])
        # miny = np.min(kutu[:, 1])
        # maxx = np.max(kutu[:, 0])
        # maxy = np.max(kutu[:, 1])

        #copy = thresh[miny:maxy, minx:maxx].copy()
        #copy = cv2.resize(copy,(500,300))
        #cv2.imshow("copy",copy)
        #cv2.waitKey()

        approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
        points = approx.ravel()
        c1 = (points[0], points[1])
        c2 = (points[2], points[3])
        c3 = (points[4], points[5])
        c4 = (points[6], points[7])

        transformed_image = perpektif(thresh, kutu)
        transformed_image1 = perpektif(im_in, kutu)
        #cv2.imshow("transformed", transformed_image)
        #cv2.waitKey(0)



        #gray_trans = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        toplam_pixeler = transformed_image.shape[0] * transformed_image.shape[1]
        siyah_pixeler = cv2.countNonZero(transformed_image)
        beyaz_pixeler = toplam_pixeler - siyah_pixeler
        oran = beyaz_pixeler / toplam_pixeler


        if (toplam_pixeler > 0):

            oran = beyaz_pixeler / toplam_pixeler
            print(f"Siyah pixel oranı: {oran:.2f}")
        if(toplam_pixeler <= 0):
            oran = 1
            print(f"Siyah pixel oranı: {oran:.2f}")

        W = maxx - minx
        H = maxy - miny
        oran2 = W/H
        oran3 = H*W
        oran4 = H/W

        print("kontur oranı:",oran2)
        print("alan: ",oran3 )

        if (16000 > oran3 > 500):
            if (oran < 0.5) and (6 > oran2 > 2):

                a = cv2.drawContours(im_in, [kutu], 0, (0, 255, 0), 2)
                cv2.imshow("contours", a)
                cv2.waitKey(0)
                cv2.imshow("transformed", transformed_image1)
                cv2.waitKey(0)
                print("plaka bulundu")

                text = pytesseract.image_to_string(transformed_image1, lang='eng')
                print("Numara: ", text)
            else:
                a = cv2.drawContours(im_in, [kutu], 0, (0, 0, 255), 2)
                cv2.imshow("contours", a)
                cv2.waitKey(0)
                cv2.imshow("transformed", transformed_image1)
                cv2.waitKey(0)
                print("plaka bulunamadı")

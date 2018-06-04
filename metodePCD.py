import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils

def rataRGB(img): #Mendapatkan rataan dari red green blue data
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Konversi Menjadi HSV
    citra_gray = cv2.cvtColor(HSV, cv2.COLOR_RGB2GRAY) #Konversi Menjadi Grayscale
    row, col, ch = img.shape #Mendapatkan ukuran row, col dan jumlah channel dari img (RGB ada 3 channel, gray cuma 1)
    sumred,sumgreen,sumblue = 0,0,0 #Set variable 0
    for k in range(0,row):
        for l in range (0,col):
            gray = citra_gray[k,l]
            blue, green, red = img[k,l]
            if(gray > 90): #Jika nilai gray lebih besar dari threshold maka nilai piksel diset 255, untuk mempersingkat hitungan
                sumred = sumred + red #Langsung dibuat jika lebih dari threshold maka dihitung nilainya (seharusnya dilakukan irisan dahulu antar kedua piksel)
                sumgreen = sumgreen + green
                sumblue = sumblue + blue

    sumred = sumred/(col*row) #Mendapatkan rata-rata
    sumgreen = sumgreen/(col*row)
    sumblue = sumblue/(col*row)

    return sumred,sumgreen,sumblue


def rataHSV(img): #Mendapatkan rataan dari red green blue data
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Konversi Menjadi HSV
    citra_gray = cv2.cvtColor(HSV, cv2.COLOR_RGB2GRAY) #Konversi Menjadi Grayscale
    row, col, ch = img.shape #Mendapatkan ukuran row, col dan jumlah channel dari img (RGB ada 3 channel, gray cuma 1)
    sumhue,sumsat,sumval = 0,0,0 #Set variable 0
    for k in range(0,row):
        for l in range (0,col):
            gray = citra_gray[k,l]
            H, S, V = HSV[k,l]
            if(gray > 90): #Jika nilai gray lebih besar dari threshold maka nilai piksel diset 255, untuk mempersingkat hitungan
                sumhue += H #Langsung dibuat jika lebih dari threshold maka dihitung nilainya (seharusnya dilakukan irisan dahulu antar kedua piksel)
                sumsat += S
                sumval += V

    sumhue /= (col*row) #Mendapatkan rata-rata
    sumsat /= (col*row)
    sumval /= (col*row)

    return sumhue,sumsat,sumval

def canny(im): #Menghitung jumlah pixel putih setelah gambar di ekstrak teksturnya menggunakan menggunakan canny
	im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(im,20,255,L2gradient=False) #Menggunakan metode canny
	hit = cv2.countNonZero(edges) #Menghitung nilai yang tidak 0
	return hit

def hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def morph(images):
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    hit = cv2.countNonZero(opening)
    return hit

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def diameterDetect(image):
    # load the image, convert it to grayscale, and blur it slightly
    global dimA,dimB
    dimA, dimB = 0,0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 0.955

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        '''
        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}in".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    
        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)
        '''
    return dimA, dimB


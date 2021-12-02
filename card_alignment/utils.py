import numpy as np
import math
from scipy.spatial import distance as dist
import cv2

def angle_between_vectors_degrees(u, v):
    return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def get_angle(p1, p2, p3):
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    avec = a - b
    cvec = c - b

    return angle_between_vectors_degrees(avec, cvec)


def angle_range(quad):
    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles) 

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their y-coordinates so we can grab the top-left and bottom-left points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an anchor to calculate the Euclidean distance between the top-left and right-most points; by the Pythagorean theorem, the point with the largest distance will be our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype = "float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def findFullLine(x, y, IM_HEIGHT, IM_WIDTH, isVertical=False ,isHorizontal=False):
    assert(isHorizontal != isVertical)
    try:
        coefficients = np.polyfit(x, y, 1)
    except:
        pass
    if isHorizontal:
        if (x[0] == x[1]):
            return ((x[0], 0), (x[0], IM_HEIGHT))
        x1 = 0
        x2 = IM_WIDTH
        y1 = int(coefficients[1])
        y2 = int(coefficients[0] * x2 + coefficients[1])
        return ((x1, y1), (x2, y2))
    if isVertical:
        if coefficients[0] == 0:
            return ((0, y[0]), (IM_WIDTH, y[0]))
        y1 = 0
        y2 = IM_HEIGHT
        x1 = int(- coefficients[1] / coefficients[0])
        x2 = int((IM_HEIGHT - coefficients[1]) / coefficients[0])
        return ((x1, y1), (x2, y2))    
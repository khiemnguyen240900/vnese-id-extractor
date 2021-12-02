import argparse
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon

from .polygon_interacter import *
from .utils import *


class CardAlignment:
    def __init__(self, interactive=False, visualization=False, result=False, MIN_QUAD_AREA_RATIO=0.15, MAX_QUAD_ANGLE_RANGE=40):
        self.interactive = interactive
        self.visualization_mode = visualization
        self.result = result
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE
        # these constants are carefully chosen
        self.MORPH = 9
        self.CANNY = 84
    
    def filter_corners(self, corners, min_dist=20):
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)
        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners
    
    def get_corners(self, img):
        lines = lsd(img)
        IM_HEIGHT, IM_WIDTH = img.shape
        corners = []
        if lines is not None:
            # separate out the horizontal and vertical lines, and draw them back onto separate canvases
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []
            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            horizontal_full_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                # cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 3)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))
                (pt1, pt2) = findFullLine([min_x, max_x], [left_y, right_y], IM_HEIGHT, IM_WIDTH, isHorizontal=True)
                cv2.line(horizontal_full_lines_canvas, pt1, pt2, 1, 1)
                
            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_full_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                # cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 3)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))
                (pt1, pt2) = findFullLine([top_x, bottom_x], [min_y, max_y], IM_HEIGHT, IM_WIDTH, isVertical=True)
                cv2.line(vertical_full_lines_canvas, pt1, pt2, 1, 1)

            # find the corners
            self.full_hv_lines = cv2.add(cv2.add(horizontal_full_lines_canvas*255, vertical_full_lines_canvas*255), cv2.add(vertical_lines_canvas*255, horizontal_lines_canvas*255))
            self.h_lines = horizontal_full_lines_canvas*255
            self.v_lines = vertical_full_lines_canvas*255
            self.hv_lines = cv2.add(horizontal_lines_canvas*255, vertical_lines_canvas*255)
            corners_y, corners_x = np.where(horizontal_full_lines_canvas + vertical_full_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # remove corners in close proximity
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

    def get_contour(self, rescaled_image):
        
        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        self.blur = gray

        # dilate helps to remove potential holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.MORPH, self.MORPH))
        dilated = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        self.dilated = dilated

        # find edges and mark them in the output map using the Canny algorithm
        edged = cv2.Canny(dilated, 0, self.CANNY)
        self.edged = edged
        
        test_corners = self.get_corners(edged)
        self.test_corners = test_corners

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)

            # get top five quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # sort candidate quadrilaterals by their angle range, which helps remove outliers
            quads = sorted(quads, key=angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            # for debugging: uncomment the code below to draw the corners and countour found 
            # by get_corners() and overlay it on the image

            # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*test_corners))
            # plt.imshow(rescaled_image)
            # plt.show()

        # also attempt to find contours directly from the edged image, which occasionally 
        # produces better results
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # If we did not find any valid contours, just use the whole image
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(('Drag the corners of the box to the corners of the document. \n'
            'Close the window when finished.'))
        p = PolygonInteractor(ax, poly, img=rescaled_image)
        plt.imshow(cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB))
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype = "int32")
        return new_points.reshape(4, 2)

    def visualization(self):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
        [axi.set_axis_off() for axi in axes.ravel()] # turn axis off
        axes[0][0].imshow(cv2.cvtColor(self.rescaled_image, cv2.COLOR_BGR2RGB))
        axes[0][1].imshow(self.blur, cmap='gray')
        axes[0][2].imshow(self.dilated, cmap='gray')
        axes[0][3].imshow(self.edged, cmap='gray')
        axes[1][0].imshow(self.full_hv_lines, cmap='gray')
        axes[1][1].imshow(self.gray, cmap='gray')
        axes[1][2].imshow(imutils.resize(self.graywarped, height=int(self.RESCALED_HEIGHT)), cmap='gray')
        axes[1][3].imshow(imutils.resize(self.thresh, height=int(self.RESCALED_HEIGHT)), cmap='gray')

        axes[1][1].scatter(*zip(*self.test_corners))
        fig.tight_layout()
        plt.show()

    def adaptiveThreshold(self, orig, screenCnt):
        warped = four_point_transform(orig, screenCnt * self.ratio)
        self.warped = warped
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self.graywarped = gray

        sharpen = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        self.thresh = thresh

    def scan(self, image):

        self.RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = 'output'

        # image = cv2.imread(image_path)

        assert(image is not None)

        self.ratio = image.shape[0] / self.RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height = int(self.RESCALED_HEIGHT))
        self.rescaled_image = rescaled_image
        screenCnt = self.get_contour(rescaled_image)
        self.adaptiveThreshold(orig, screenCnt)

        # if self.visualization_mode:
        #     self.visualization()
        # if self.interactive:
        #     screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)
        #     self.adaptiveThreshold(orig, screenCnt)
        # if self.result:
            # plt.imshow(cv2.cvtColor(self.graywarped, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()

        return cv2.cvtColor(self.warped, cv2.COLOR_BGR2RGB)

        # basename = os.path.basename(image_path)
        # basetype = basename.split('-')[0]
        # cv2.imwrite(OUTPUT_DIR + '/binary/' + basetype + '/' + basename, self.thresh)
        # cv2.imwrite(OUTPUT_DIR + '/gray/' + basetype + '/' + basename, self.graywarped)
        # cv2.imwrite(OUTPUT_DIR + '/color/' + basetype + '/' + basename, self.warped)
        # print("Proccessed " + basename)
    
    def scan_card(self, image, FLAGS=None):

        self.RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = 'output'

        assert(image is not None)

        self.ratio = image.shape[0] / self.RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height = int(self.RESCALED_HEIGHT))
        self.rescaled_image = rescaled_image
        screenCnt = self.get_contour(rescaled_image)
        self.adaptiveThreshold(orig, screenCnt)

        if FLAGS.alignment_process:
            self.visualization()
        if FLAGS.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)
            self.adaptiveThreshold(orig, screenCnt)
        # if self.result:
            # plt.imshow(cv2.cvtColor(self.graywarped, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()

        #return cv2.cvtColor(self.warped, cv2.COLOR_BGR2RGB)
        return self.warped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true',
        help = "Flag for manually verifying and/or setting document corners")
    ap.add_argument("-v", action='store_true',
        help = "Flag for visualizing the process")
    ap.add_argument("-r", action='store_true',
        help = "Flag for visualizing the result")

    args = vars(ap.parse_args())
    im_dir = args["images"]
    im_file_path = args["image"]
    interactive_mode = args["i"]
    visualization_mode = args["v"]
    result = args["r"]

    scanner = CardAlignment(interactive_mode, visualization_mode, result)

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        scanner.scan(im_file_path)

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
        for im in im_files:
            scanner.scan(im_dir + '/' + im)
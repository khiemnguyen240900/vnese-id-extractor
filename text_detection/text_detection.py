import cv2
from .utils.image_utils import sort_text
from .detector import Detector

class TextDetection(object):
    def __init__(self, path_model, path_labels, thres_nms, thres_score):
        self.text_detection_model = Detector(path_to_model=path_model,
                                             path_to_labels=path_labels,
                                             nms_threshold=thres_nms,
                                             score_threshold=thres_score)
        # init boxes
        self.id_boxes = None
        self.name_boxes = None
        self.birth_boxes = None
        self.add_boxes = None
        self.home_boxes = None

    def predict(self, img):
        # detect text boxes
        image = cv2.resize(img, (900, 600))
        detection_boxes, detection_classes, category_index = self.text_detection_model.predict(image)

        # sort text boxes according to coordinate
        self.id_boxes, self.name_boxes, self.birth_boxes, self.home_boxes, self.add_boxes = sort_text(detection_boxes,
                                                                                                      detection_classes)
        return detection_boxes, detection_classes, category_index
    
    def draw(self, img):
        return self.text_detection_model.draw(img)
    

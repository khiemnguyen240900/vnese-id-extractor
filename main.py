import os
import sys
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS

sys.path.insert(0, './yolov4_card_detection')


from model_config import text_detection_config
from yolov4_card_detection.card_detection import detect_card
from card_alignment.card_alignment import CardAlignment
from text_detection.text_detection import TextDetection
from text_recognition.text_recognition import TextRecognition

from text_detection.utils.image_utils import sort_text

import cv2
import numpy as np
import json
import codecs

# Flag for phone camera
flags.DEFINE_string('camera_ip', None, 'camera ip')

# Flag for card detection
flags.DEFINE_string('weights', '/checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.70, 'score threshold')

# Flag for output
flags.DEFINE_boolean('output', False, 'to save aligned image to output folder')

# Flag for card alignment
flags.DEFINE_boolean('interactive', False, 'interactive mode in card alignment')
flags.DEFINE_boolean('alignment_process', False, 'show alignment process')

def main(_argv):
    text_detection_model = TextDetection(path_model=text_detection_config['path_to_model'],
                                                path_labels=text_detection_config['path_to_labels'],
                                                thres_nms=text_detection_config['nms_ths'], 
                                                thres_score=text_detection_config['score_ths'])
    aligned_model = CardAlignment()
    text_recognition_model = TextRecognition()
    
    # detect card type
    frame = detect_card(FLAGS)
    
    # align card
    aligned = aligned_model.scan_card(frame, FLAGS)

    # detect text
    detected = np.copy(aligned)
    detection_boxes, detection_classes, category_index = text_detection_model.predict(detected)
    id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes = sort_text(detection_boxes, detection_classes)
    detected = text_detection_model.draw(detected)

    # recognize text
    field_dict = text_recognition_model.recog_text(aligned, id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes)
    
    # save extracted information
    output_dir = os.getcwd().replace(os.sep, '/') + "/output"
    with codecs.open(output_dir + "/information.txt", "w", "utf-8") as file:
        file.write(json.dumps(field_dict,ensure_ascii=False)) # use `json.loads` to do the reverse

    # save aligned image
    if FLAGS.output:
        cv2.imwrite(output_dir + "/aligned.jpg", aligned)
        cv2.imwrite(output_dir + "/detected.jpg", detected)

    cv2.imshow("Alignment", aligned)
    cv2.imshow("Detected", detected)
    cv2.waitKey(0)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
import os
import cv2
import tensorflow as tf
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')

PATH_CONFIG=r"C:\Users\alan-\Tensorflow\workspace\training_demo\exported-models\my_model\pipeline.config"
CHECKPOINT_PATH=r"C:\Users\alan-\Tensorflow\workspace\training_demo\models\my_ssd_resnet101_v1_fpn"
PATH_TO_LABELS=r"C:\Users\alan-\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt"

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
	
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_CONFIG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-8')).expect_partial()

@tf.function
def detect_fn(image):
	"""Detect objects in image."""
	image, shapes = detection_model.preprocess(image)
	prediction_dict = detection_model.predict(image, shapes)
	detections = detection_model.postprocess(prediction_dict, shapes)
	return detections, prediction_dict, tf.reshape(shapes, [-1])
    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, prediction_dict, shapes  = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
                  
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0,
        agnostic_mode=False)
        
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

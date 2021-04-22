from generators.csv_ import CSVGenerator
from models.resnet import centernet
import cv2
import os
import numpy as np
import time
from generators.utils import affine_transform, get_affine_transform
from utils.image import read_image_bgr, preprocess_image, resize_image
import os.path as osp

DATA_SUFFIX = '_datamap.png'
RESULT_PATH = "result/"
PROCESS_PATH = "process/"
model_path = 'checkpoints/csv.h5'
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
score_threshold = 0.5
flip_test = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
generator = CSVGenerator(
    'data/annotations.csv',
    'data/classes.csv',
    'data',
)

num_classes = generator.num_classes()
classes = list(generator.classes.keys())

model, prediction_model, debug_model = centernet(num_classes=num_classes,
                                                 nms=True,
                                                 flip_test=flip_test,
                                                 freeze_bn=False,
                                                 score_threshold=score_threshold)
prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)


for f in os.listdir(PROCESS_PATH):
    if f.endswith(DATA_SUFFIX):
        image = read_image_bgr(PROCESS_PATH + f)
        src_image = image.copy()

        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0

        tgt_w = generator.input_size
        tgt_h = generator.input_size
        
        trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
        image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
        image = image.astype(np.float32)

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68
        
        print(image.shape)
        
        if flip_test:
            flipped_image = image[:, ::-1]
            inputs = np.stack([image, flipped_image], axis=0)
        else:
            inputs = np.expand_dims(image, axis=0)
        # run network
        start = time.time()
        detections = prediction_model.predict_on_batch(inputs)[0]
        print(time.time() - start)
        scores = detections[:, 4]
        # select indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]

        # select those detections
        detections = detections[indices]
        detections_copy = detections.copy()
        detections = detections.astype(np.float64)
        trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

        for j in range(detections.shape[0]):
            detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
            detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])
        
        with open(RESULT_PATH + f[:-len(DATA_SUFFIX)] + '.txt', 'w') as output:
            for detection in detections:
                xmin = int(round(detection[0]))
                ymin = int(round(detection[1]))
                xmax = int(round(detection[2]))
                ymax = int(round(detection[3]))
                score = '{:.4f}'.format(detection[4])
                class_id = int(detection[5])
                
                print(f'{xmin},{ymin},{xmax},{ymax},{class_id}', file=output)
                
                color = colors[class_id]
                class_name = classes[class_id]
                label = '-'.join([class_name, score])
                ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
                cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 4)
                cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
                cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            cv2.imwrite(RESULT_PATH + f[:-len(DATA_SUFFIX)] + '_result.png', src_image)
            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.imshow('image', src_image)
            #key = cv2.waitKey(0)

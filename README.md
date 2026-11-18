# En-YOLO: An Implicit-Aware Object Detector Centered under Low-Light Enhancement

This repository contains the En-YOLO source code and class lists for the ExDark, VOC, and COCO datasets.

- `train.py` trains the model. Set the annotation and dataset paths for your local environment. The training script expects `model_data/yolox_s.pth` for pretrained initialization.
- `predict.py` runs inference. The default checkpoint path in `yolo.py` is `logs/ep200-loss0.353-val_loss2.601.pth`.
- `exdark_get_map.py` evaluates detections on ExDark. Set the dataset path before running it.

See [LICENSE-YOLOX](LICENSE-YOLOX) for the upstream YOLOX license.

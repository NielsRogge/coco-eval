# coco-eval (PyTorch)
A tiny package supporting distributed computation of COCO metrics (like mAP) for PyTorch models.

## Installation

I made this package available on PyPi (thanks to [this guide](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)): https://pypi.org/project/coco-eval/. 

```
pip install coco-eval
```

## Usage

The metric was taken and isolated from the [DETR repository](https://github.com/facebookresearch/detr/tree/main). Credits go to the authors.

Usage is as follows:

```
from coco_eval import CocoEvaluator
from torchvision.datasets import CocoDetection

dataset = CocoDetection(root="path_to_your_images", annFile="path_to_annotation_file")

evaluator = CocoEvaluator(coco_gt=dataset.coco, iou_types=["bbox"])

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
```

Refer to my DETR fine-tuning demo notebook regarding an example of using it.

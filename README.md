# coco-eval (PyTorch)
A tiny package supporting distributed computation of COCO metrics (like mAP) for PyTorch models.

## Installation

I made this package available on PyPi (thanks to [this guide](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)): https://pypi.org/project/coco-eval/. 

```
pip install coco-eval
```

## Usage

The metric was taken and isolated from the [DETR repository](https://github.com/facebookresearch/detr/tree/main). Credits go to the authors.

High-level usage is as follows (assuming you have a PyTorch model that makes predictions):

```
from coco_eval import CocoEvaluator
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

dataset = CocoDetection(root="path_to_your_images", annFile="path_to_annotation_file")

dataloader = DataLoader(dataset, batch_size=2)

evaluator = CocoEvaluator(coco_gt=dataset.coco, iou_types=["bbox"])

model = ...

for batch in dataloader:
   predictions = model(batch)
   
   evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
```

Refer to my [DETR fine-tuning demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb) regarding an example of using it.

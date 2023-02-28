# coco-eval
A tiny package supporting distributed computation of COCO metrics.

## Usage

The metric was isolated from the DETR repository. Usage is as follows:

```
from coco_eval import CocoEvaluator

evaluator = CocoEvaluator(coco_gt=..., iou_types=["bbox"])

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
```

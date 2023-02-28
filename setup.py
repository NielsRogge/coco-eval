import setuptools

setuptools.setup(
    name="coco_eval",
    version="0.0.1",
    author="Niels Rogge",
    author_email="niels.rogge1@gmail.com",
    description="A tiny package containing a COCO evaluator that works in distributed environments.",
    install_requires=[
        'numpy',
        'torch',
        'pycocotools',
      ],
)
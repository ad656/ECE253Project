# Goal

Improved Anomaly detection system. We aim to improve anomaly and objection detection in images, specifically those taken in conditions with overexposure or haze.
We do this by applying algorithms to pre-process the initial image before running the objection detection system. 


## Installation

Clone the repository

```bash
git clone https://github.com/ad656/ECE253Project.git
```

## Image preprocess

Run following commands to get fusion result of your images.

```bash
cd ECE253Project
python3 fusion.py —input_dir your_images_dir —output target_dir
```
Use dehaze parameter to select dcp for dcp and hazeline for hazeline.

## Run anomaly detection

```bash
cd ECE253Project/detecting_the_unexpected
```
Put your target images in ```ECE253Project/detecting_the_unexpected/data/my_test_data``` and run python notebook Exec_Joint_pipeline.ipynb for discrepancy map.

You might need to change the extension in ``` detecting_the_unexpected/src/datasets/dataset.py ``` line 538 to get your images.

For ROCs and AUC, put your images, labels in ```detecting_the_unexpected/datasets/dataset_RoadAnomaly/frames``` and one image list in ```detecting_the_unexpected/datasets/dataset_RoadAnomaly``` and run python notebook Exec_Evaluations.ipynb

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Experiments
#### Ablation study in dataset variables

below result test in all 4077 images
mAP / F1
| Variables                   | RetinaNet     | Faster RCNN   | DirtNet       |
| --------------------------- | ------------- | ------------- | ------------- |
| baseline                    | 0.203 / 0.240 | 0.193 / 0.248 |               |
| + diff blendings            | 0.227 / 0.279 | 0.214 / 0.271 |               |
| + whole image argumentation | 0.477 / 0.507 | 0.552 / 0.570 |               |
| + fg argumentation          | 0.617 / 0.587 | 0.647 / 0.651 | 21h 20min     |
| + office items              | 0.633 / 0.607 | 0.652 / 0.659 | 0.720 / 0.791 |


#### Ablation study on model performance and complexity across different optimizations

input resolution 512 x 640

| improvements                               | mAP   | F1    | GFLOPs | params (M) | model size | infer. time    |
| ------------------------------------------ | ----- | ----- | ------ | ---------- | ---------- | -------------- |
| Baseline                                   | 0.633 | 0.607 | 71.77  | 35.64      | 147 MB     | 0.0471 s / img |
| + single-class model (1:1 random sampling) | 0.802 | 0.721 | 66.51  | 33.59      | 139 MB     | 0.0301 s / img |
| + reg FPs bbox                             | 0.798 | 0.723 | 66.51  | 33.59      | 139 MB     | 0.0301 s / img |
| + SSH layer                                | 0.840 | 0.771 | 57.26  | 29.97      | 126 MB     | 0.0238 s / img |
| + reduce channel                           | 0.825 | 0.779 | 53.82  | 24.77      | 97 MB      | 0.0184 s / img |
| + backbone(dla60xc)                        | 0.771 | 0.739 | 8.53   | 1.84       | 9.2 MB     | 0.0228 s / img |

#### Comparison of performance and training time
90 -> 91
| Models                  | P     | R     | mAP   | F1    | training time     |
| ----------------------- | ----- | ----- | ----- | ----- | ----------------- |
| Faster RCNN             | 0.658 | 0.717 | 0.677 | 0.644 | 7h 06min +- 30min |
| Faster RCNN (init)      | 0.709 | 0.685 | 0.657 | 0.645 | 5h 30min +- 35min |
| RetinaNet               | 0.668 | 0.712 | 0.660 | 0.642 | 6h 53min +- 33min |
| RetinaNet (init)        | 0.665 | 0.707 | 0.640 | 0.606 | 5h 09min +- 32min |
| resnet50 (single model) | 0.747 | 0.902 | 0.844 | 0.791 | 32min +- 8min     |
| dla60xc (single model)  | 0.731 | 0.851 | 0.815 | 0.777 | 37min +- 9min     |


#### Comparison between different sampling strategies (test in InstanceNet(dla60xc))

| sampling method                    | Precision | Recall | mAP   | F1    | Training time  |
| ---------------------------------- | --------- | ------ | ----- | ----- | -------------- |
| 0 fp samples                       | 0.013     | 0.942  | 0.243 | 0.025 | 22min +- 12min |
| 1:1 random sampling                | 0.694     | 0.819  | 0.766 | 0.739 | 33min +- 8min  |
| 1:3 random sampling                | 0.723     | 0.809  | 0.795 | 0.745 | 51min +- 9min  |
| 1:1 dynamic sampling               | 0.731     | 0.851  | 0.815 | 0.777 | 37min +- 9min  |
| total init from most similar model | 0.696     | 0.788  | 0.721 | 0.719 | 35min +- 11min |


#### Model update frequency & improment when retraining
threshold of avg_prob for update the existing models: 0.15
| scenario | updated models number | update one old model time | train new model time | mAP/F1 (new instance) | mAP/F1 existing model improvement |
| -------- | --------------------- | ------------------------- | -------------------- | --------------------- | --------------------------------- |
| 20 -> 21 | 0.7 +- 0.4            | 12 +- 6 min               | 33 +- 9 min          | 76.1 / 55.4           | 74.2/64.6 -> 76.7/70.0            |
| 50 -> 51 | 1.7 +- 1.2            | 13 +- 7 min               | 34 +- 11 min         | 74.6 / 66.9           | 73.1/65.3 -> 75.3/70.2            |
| 90 -> 91 | 2.2 +- 1.2            | 13 +- 6 min               | 37 +- 9 min          | 81.5 / 77.7           | 77.1/73.9 -> 78.5/74.8            |

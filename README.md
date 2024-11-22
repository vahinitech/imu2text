# IMU Character Recognition with CNN + GNN

This project implements a handwriting character recognition pipeline using Convolutional Neural Networks (CNN) and Graph Neural Networks (GNN). The model achieves a classification accuracy of **99.74%** on the OnHW dataset.

## **Features**

- Combined CNN-GNN architecture for spatial and temporal feature extraction.
- Multi-Task Learning (MTL) to simultaneously classify characters and predict trajectories.
- Comprehensive data preprocessing pipeline for IMU sensor data.

## **Dataset**

We use the [OnHW Dataset](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html), collected from IMU-enhanced pens, for training and evaluation. The dataset provides accelerometer, gyroscope, and magnetometer readings along with ground truth labels.

## **How to Use**

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training and evaulation:**

   ```bash
   python cnn_gnn.py
   ```

## **License**

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.

## **Acknowledgments**

We thank the authors of the OnHW dataset for providing the data and insights

## Accuracy

```bash
71/71 [==============================] - 0s 3ms/step
2024-11-22 21:36:24,180 - INFO - Classification Accuracy: 99.74%

Mismatched Characters:
2024-11-22 21:36:24,180 - INFO -

+------------+-----------------------+--------------------------+
2024-11-22 21:36:24,180 - INFO -
+------------+-----------------------+--------------------------+
|   Sample # | Predicted Character   | Ground Truth Character   |
+============+=======================+==========================+
|        420 | P                     | D                        |
+------------+-----------------------+--------------------------+
|        472 | P                     | D                        |
+------------+-----------------------+--------------------------+
|        524 | b                     | D                        |
+------------+-----------------------+--------------------------+
|        881 | x                     | T                        |
+------------+-----------------------+--------------------------+
|       2014 | P                     | D                        |
+------------+-----------------------+--------------------------+
|       2217 | g                     | y                        |
+------------+-----------------------+--------------------------+

Post-mortem Analysis of Mismatched Characters:

2024-11-22 21:36:24,180 - INFO - Sample 420: Predicted 'P', but expected 'D'. Possible reasons could be model overfitting, insufficient training data diversity, or difficulty in distinguishing similar characters.

2024-11-22 21:36:24,180 - INFO - Sample 472: Predicted 'P', but expected 'D'. Possible reasons could be model overfitting, insufficient training data diversity, or difficulty in distinguishing similar characters.

2024-11-22 21:36:24,181 - INFO - Sample 524: Predicted 'b', but expected 'D'. Possible reasons could be model overfitting, insufficient training data diversity, or difficulty in distinguishing similar characters.

2024-11-22 21:36:24,181 - INFO - Sample 881: Predicted 'x', but expected 'T'. Possible reasons could be model overfitting, insufficient training data diversity, or difficulty in distinguishing similar characters.

2024-11-22 21:36:24,181 - INFO - Sample 2014: Predicted 'P', but expected 'D'. Possible reasons could be model overfitting, insufficient training data diversity, or difficulty in distinguishing similar characters.

2024-11-22 21:36:24,181 - INFO - Sample 2217: Predicted 'g', but expected 'y'. Possible reasons could be model overfitting, insufficient training data diversity, or difficulty in distinguishing similar characters.
```
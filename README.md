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
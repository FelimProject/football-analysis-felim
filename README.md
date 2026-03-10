### <h1>Football Match Outcome Prediction Using Yolo 11</h1>

Football Analysis is a data-driven project designed to extract meaningful insights from football match statistics, team performance, and historical match data. This project is suitable for data analysts, football enthusiasts, and developers who want to build analytical dashboards or predictive models for football match outcomes.

The proposed model utilizes machine learning techniques to analyze historical match data and predict the possible outcomes of football matches, such as win, loss, or draw. By learning patterns from team statistics, player performance, and match history, the model aims to assist in football analytics and decision-making

## Model Evaluation Results

The trained object detection model was evaluated using the validation dataset to measure its detection performance across different classes. The model architecture consists of 190 layers with a total of 56,831,644 parameters and approximately 194.4 GFLOPs of computational complexity.

## Overall Model Performance

| Metric | Score |
|------|------|
| Precision | 0.893 |
| Recall | 0.720 |
| mAP@50 | 0.786 |
| mAP@50-95 | 0.531 |

## Class-wise Detection Performance

| Class | Precision | Recall | mAP@50 |
|------|------|------|------|
| Player | 0.943 | 0.948 | 0.981 |
| Goalkeeper | - | - | 0.896 |
| Referee | - | - | 0.915 |
| Ball | - | - | 0.352 |

Pretrained Model

The pretrained model can be accessed at the following link:

https://drive.google.com/file/d/1bZmV9nCr1hLJkHB7BduRZapWNLJVOcFL/view?usp=drive_link

This model can be used directly for inference or further fine-tuned for additional football analysis tasks.

Overall, the model demonstrates strong capability in detecting major football match entities such as players,balls and referees, while further improvements may be required to enhance ball detection accuracy.

## Dataset Description

This project utilizes the **football players detection dataset** available on Roboflow Universe. The dataset can be accessed through the following link:

Dataset Documentation:  
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

The dataset is designed for computer vision tasks in football match analysis and contains annotated images that enable object detection models to identify key entities during a match. The dataset includes approximately **372 annotated images** and supports multiple object detection architectures such as YOLO and RF-DETR.

Four main object classes are annotated in the dataset:

- **Player**
- **Referee**
- **Ball**

Each image contains bounding box annotations for these objects, allowing machine learning models to learn spatial features and object characteristics. This dataset is commonly used for training object detection models to analyze football matches, including player tracking, tactical analysis, and automated broadcast analytics.

The dataset is released under the **CC BY 4.0 license**, meaning it can be used for research and development purposes with proper attribution to the original creators. 


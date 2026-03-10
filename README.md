### <h1>Football Match Outcome Prediction Using Yolo 11</h1>

Football Analysis is a data-driven project designed to extract meaningful insights from football match statistics, team performance, and historical match data. This project is suitable for data analysts, football enthusiasts, and developers who want to build analytical dashboards or predictive models for football match outcomes.

The proposed model utilizes machine learning techniques to analyze historical match data and predict the possible outcomes of football matches, such as win, loss, or draw. By learning patterns from team statistics, player performance, and match history, the model aims to assist in football analytics and decision-making

## Model Evaluation Results

The trained object detection model was evaluated using the validation dataset to measure its detection performance across different classes. The model architecture consists of 190 layers with a total of 56,831,644 parameters and approximately 194.4 GFLOPs of computational complexity.

## Model Evaluation Results

The trained YOLOv11 object detection model was evaluated using the validation dataset to measure detection performance across different classes.

### Overall Model Performance

| Dataset | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|--------|-----------|-----------|--------|--------|-----------|
| Validation Set | 38 | 905 | 0.893 | 0.720 | 0.786 | 0.531 |

### Class-wise Detection Performance

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|------|------|------|------|------|------|------|
| Ball | 35 | 35 | 1.000 | 0.247 | 0.352 | 0.150 |
| Goalkeeper | 27 | 27 | 0.792 | 0.815 | 0.896 | 0.639 |
| Player | 38 | 754 | 0.943 | 0.948 | 0.981 | 0.757 |
| Referee | 38 | 89 | 0.838 | 0.870 | 0.915 | 0.579 |

### Performance Analysis

The model demonstrates strong detection capability for the **player class**, achieving the highest performance with an **mAP@50 of 0.981**, indicating highly reliable detection accuracy. The **goalkeeper** and **referee** classes also show strong results with **mAP@50 scores of 0.896 and 0.915**, respectively.

However, the **ball class** exhibits lower detection performance with an **mAP@50 of 0.352** and **mAP@50-95 of 0.150**, mainly due to the **small object size**, **motion blur**, and **limited training samples** in the dataset.

The pretrained model can be accessed at the following link:

https://drive.google.com/file/d/1bZmV9nCr1hLJkHB7BduRZapWNLJVOcFL/view?usp=drive_link

This model can be used directly for inference or further fine-tuned for additional football analysis tasks.

Overall, the model demonstrates strong capability in detecting major football match entities such as players,balls, Goalkeeper and referees, while further improvements may be required to enhance ball detection accuracy.

## Dataset Description

This project utilizes the **football players detection dataset** available on Roboflow Universe. The dataset can be accessed through the following link:

Dataset Documentation:  
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

The dataset is designed for computer vision tasks in football match analysis and contains annotated images that enable object detection models to identify key entities during a match. The dataset includes approximately **372 annotated images** and supports multiple object detection architectures such as YOLO and RF-DETR.

Four main object classes are annotated in the dataset:

- **Player**
- **Referee**
- **Ball**
- **Goal keeper**

Each image contains bounding box annotations for these objects, allowing machine learning models to learn spatial features and object characteristics. This dataset is commonly used for training object detection models to analyze football matches, including player tracking, tactical analysis, and automated broadcast analytics.

The dataset is released under the **CC BY 4.0 license**, meaning it can be used for research and development purposes with proper attribution to the original creators. 

## Demo Video
<img width="1233" height="697" alt="image" src="https://github.com/user-attachments/assets/75000fba-3edc-400e-abd7-bd3f6bc523c4" />

[Watch Demo Video](https://drive.google.com/file/d/1w3PPwaI7dKFlkz6zOabVGfAO8iaJ2gRH/view)


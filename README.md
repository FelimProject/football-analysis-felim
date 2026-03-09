### <h1>Football Match Outcome Prediction Using Yolo 11</h1>

Football Analysis is a data-driven project designed to extract meaningful insights from football match statistics, team performance, and historical match data. This project is suitable for data analysts, football enthusiasts, and developers who want to build analytical dashboards or predictive models for football match outcomes.

The proposed model utilizes machine learning techniques to analyze historical match data and predict the possible outcomes of football matches, such as win, loss, or draw. By learning patterns from team statistics, player performance, and match history, the model aims to assist in football analytics and decision-making

## Model Evaluation Results

The trained object detection model was evaluated using the validation dataset to measure its detection performance across different classes. The model architecture consists of 190 layers with a total of 56,831,644 parameters and approximately 194.4 GFLOPs of computational complexity.

Overall, the model achieved strong detection performance with a precision score of **0.893**, recall of **0.720**, and a mean Average Precision (mAP@50) of **0.786**, while the stricter metric mAP@50-95 reached **0.531**.

Class-wise evaluation shows that the model performs particularly well in detecting **players**, achieving a precision of **0.943**, recall of **0.948**, and mAP@50 of **0.981**, indicating highly reliable detection performance. The **goalkeeper** class also shows strong results with an mAP@50 of **0.896**.

Detection performance for **referees** is moderate with an mAP@50 of **0.915**, while the **ball** class shows lower performance due to its small object size and fewer training instances, achieving an mAP@50 of **0.352**.

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

# Football Match Outcome Prediction Using YOLOv11

**Football Analysis** is a data-driven project designed to extract meaningful insights from football match statistics, team performance, and historical match data. This project is ideal for data analysts, football enthusiasts, and developers who want to build analytical dashboards or predictive models for football match outcomes.

The proposed model utilizes machine learning techniques to analyze historical match data and predict possible outcomes of football matches (win, loss, or draw). By learning patterns from team statistics, player performance, and match history, the model assists in football analytics and decision-making.

---

## Model Architecture

The YOLOv11 object detection model used in this project consists of:

- **Layers:** 190
- **Parameters:** 56,831,644
- **Computational Complexity:** ~194.4 GFLOPs

The model detects four main entities in football matches: **Player, Referee, Ball, Goalkeeper**.

---

## Model Evaluation Results

The model was evaluated on the validation dataset. Below is a summary:

### Overall Performance

| Dataset          | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-----------------|-------|-----------|-----------|--------|--------|-----------|
| Validation Set   | 38    | 905       | 0.893     | 0.720  | 0.786  | 0.531     |

### Class-wise Detection Performance

| Class        | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------------|--------|-----------|-----------|--------|--------|-----------|
| Ball         | 35     | 35        | 1.000     | 0.247  | 0.352  | 0.150     |
| Goalkeeper   | 27     | 27        | 0.792     | 0.815  | 0.896  | 0.639     |
| Player       | 38     | 754       | 0.943     | 0.948  | 0.981  | 0.757     |
| Referee      | 38     | 89        | 0.838     | 0.870  | 0.915  | 0.579     |

**Analysis:**

- The **Player** class achieves the highest detection performance (mAP@50 = 0.981).  
- Goalkeeper and Referee classes also show strong results.  
- The **Ball** class shows lower performance due to small object size, motion blur, and limited training data.

---

## Pretrained Model

The pretrained YOLOv11 model can be accessed here:

[Download Model](https://drive.google.com/file/d/1bZmV9nCr1hLJkHB7BduRZapWNLJVOcFL/view?usp=drive_link)

The model can be used for **direct inference** or **further fine-tuning** for football analysis tasks.

---

## Dataset Description

This project uses the **Football Players Detection Dataset** from Roboflow Universe:

[Dataset Documentation](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)

- **Images:** ~372 annotated images  
- **Object Classes:** Player, Referee, Ball, Goalkeeper  
- **License:** CC BY 4.0 (free for research & development with attribution)

Each image contains bounding box annotations for object detection, enabling models to learn spatial features and object characteristics. This dataset is widely used for tasks like **player tracking**, **tactical analysis**, and **automated broadcast analytics**.

---

## Demo Video

[![Watch Demo Video](thumbnail.png)](https://drive.google.com/file/d/1bZmV9nCr1hLJkHB7BduRZapWNLJVOcFL/view?usp=drive_link)

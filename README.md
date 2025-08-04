Here's a polished and professional `README.md` for your **T.R.A.C.K. – Drone Trash Detection Application**. This version is designed for GitHub and showcases your system’s purpose, features, architecture, and technical decisions clearly and concisely.

---

# ♻️ T.R.A.C.K. – Trash Recognition and Collection Keeper

**T.R.A.C.K.** is a drone-powered trash detection system that uses deep learning and computer vision to identify, segment, and quantify waste in real time from aerial imagery. It’s designed for environmental cleanup initiatives, offering a scalable and intelligent solution for trash monitoring in public, remote, or hard-to-reach areas.

---

## 🌍 Motivation

Over 2 billion tons of municipal solid waste are generated globally each year, with one-third not managed in an environmentally safe way. T.R.A.C.K. addresses this challenge using AI and drone technology to:

* Automate litter detection and quantification
* Provide real-time insights for environmental cleanup efforts
* Enable future capabilities like autonomous trash collection

---

## 🧠 Features

* ✅ **Real-time Trash Detection** using YOLOv8 instance segmentation models
* 📷 **Drone Video Feed Integration** with overlayed detections
* 📊 **Volume Estimation** of detected trash using bounding box heuristics
* 📈 **Data Visualization** of trash distribution (count & volume per type)
* 💾 **Data Export** to CSV and PNG for further analysis or reporting
* 🧪 **Modular GUI** built with Tkinter for easy operation and analysis

---

## 🖥️ Application Architecture

### 🔄 System Flow

1. Initialize drone and video feed
2. Detect trash via YOLOv8 model
3. Optionally adjust confidence threshold
4. Capture and log frames using "Screenshot" functionality
5. Visualize detected trash stats (volume, type, count)
6. Export data and images

### 📷 CNN Model

* **YOLOv8 Nano (YOLOv8n)**: Lightweight, anchor-free, and optimised for edge performance.
* Optimised via **hyperparameter tuning** and **class balancing** to address real-world imbalances in the dataset.

---

## 🖼️ GUI Overview

| Main Dashboard                                                                                                                                                         | Settings                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 📺 Live video feed with detection overlays <br> 🖼 Screenshot capture + removal <br> 📊 Trash summary table <br> 📈 Visualization: Bar plots (count & volume per type) | 🎚 Confidence slider (0–1) <br> 📁 Load YOLO models <br> 💾 Export CSV/plots <br> ❌ Exit app |

---

## 📦 Dataset

**[TACO – Trash Annotations in Context](https://tacodataset.org/)**

* 1,500+ annotated real-world images across 60 litter classes
* Used for training and evaluation
* Preprocessed into YOLO format, stratified and oversampled to address class imbalance

---

## 📊 Experimental Evaluation

* Evaluated **YOLOv11n**, **YOLOv8n**, and **YOLOv8s**
* **YOLOv8n** selected for best trade-off between performance and efficiency
* Achieved improvements through:

  * Stratified sampling & oversampling
  * Hyperparameter tuning (fitness-based search)
  * Label smoothing, DFL, and adaptive learning rate
* Final metrics:

  * **mAP\@50**: \~0.35
  * **mAP\@50–95**: \~0.30
  * Noted improvement in model generalization and prediction stability

---

## 🧪 Limitations & Future Work

### 📉 Current Limitations

* Class imbalance (e.g., underrepresented classes like gloves or shoes)
* Inconsistent segmentation quality in the dataset
* Challenges detecting very small or occluded objects

### 🚀 Future Extensions

* Integrate drone control and trash collection modules
* Improve GUI with:

  * Real-time inference rate tuning
  * Editable annotations
  * Cross-platform support
* Explore ensemble models or architectural improvements

---

## 🛠️ Tech Stack

| Component        | Technology                      |
| ---------------- | ------------------------------- |
| Object Detection | YOLOv8 (Ultralytics)            |
| Dataset          | TACO                            |
| GUI              | Tkinter                         |
| Backend          | Python, OpenCV, CSV, Matplotlib |
| Training         | PyTorch, Ultralytics CLI        |

---

## 👥 Contributors

* **Tom Golding** – Model training, preprocessing, tuning, evaluation, documentation
* **Dimitar Ivanov Dimitrov** – GUI development, integration, visualizations, presentation
* **Team:** *The Garbage Men* | **Project #:** 37

---

## 📄 References

* [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
* [TACO Dataset (Proença & Simões, 2020)](https://arxiv.org/abs/2003.06975)
* [World Bank: Solid Waste Management](https://datatopics.worldbank.org/what-a-waste/trends_in_solid_waste_management.html)

---

Let me know if you'd like a version with screenshots, badges, or setup instructions for running the GUI locally.

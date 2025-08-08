# 🛣️ Lane Detection Using Deep Learning

## 📌 Overview
This project presents a **deep learning-based lane detection system** designed to accurately identify lane markings in road images and videos.  
The primary aim is to assist **Advanced Driver Assistance Systems (ADAS)** by providing robust lane detection in various driving conditions, enhancing road safety and driver awareness.

## 🖼 Demo Results
| Detected Lane Example 1 | Detected Lane Example 2 |
|-------------------------|-------------------------|
| ![Lane Detection Result 1](demo_images/image1.png) | ![Lane Detection Result 2](demo_images/image2.png) |

## ⚙ How It Works
1. **Data Acquisition** – Lane detection datasets with annotated lane markings were used.
2. **Preprocessing** –  
   - Resizing frames  
   - Normalization  
   - Noise reduction and edge enhancement  
3. **Deep Learning Model** –  
   - CNN-based semantic segmentation  
   - Trained on annotated lane images to generate binary lane masks  
4. **Post-processing** –  
   - Lane contour extraction  
   - Overlay of detected lanes on the original frame  
5. **Output** – Continuous real-time lane marking visualization.

## 🛠 Tech Stack
- **Python** – Programming language
- **OpenCV** – Image and video processing
- **TensorFlow / Keras** – Deep learning framework
- **NumPy / Pandas** – Data manipulation
- **Matplotlib** – Visualization

# CourtsideAI

## On-Court Player Recognition Using Computer Vision

CourtsideAI is a computer vision project that utilizes deep learning to recognize and classify basketball players on the court. Using **MobileNetV2** and **PyTorch**, this model processes images, applies transformations, and trains on a dataset of labeled basketball players to achieve accurate player recognition.

---

## Features

- **Pretrained MobileNetV2 Backbone**: Efficient and accurate deep learning model.
- **Custom Dataset Handling**: Automatically processes images and removes corrupted files.
- **Advanced Image Augmentation**: Includes cropping, rotation, flipping, color jitter, and blurring.
- **Weighted Loss Function**: Balances class distribution for better performance.
- **Training & Validation Pipelines**: Includes accuracy tracking and early stopping.
- **Confusion Matrix & Performance Metrics**: Visualizes model effectiveness.

---

## Installation

### Prerequisites

Ensure you have Python installed (>=3.8) along with the required dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow
```

### Clone Repository

```bash
git clone https://github.com/yourusername/CourtsideAI.git
cd CourtsideAI
```

---

## Dataset Preparation

1. **Organize Images**: Place images in `images/simple_images/` following the `ImageFolder` structure:
   ```
   images/
   ├── simple_images/
   │   ├── player1/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   ├── player2/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   ```
2. **Preprocess Data**: Run the script to clean and preprocess images.

---

## Training the Model

Run the following command to train the model:

```bash
python train.py
```

- The model saves the best weights automatically.
- Training metrics and confusion matrix are displayed at the end.

---

## Model Evaluation

To evaluate the trained model, run:

```bash
python evaluate.py
```

This script generates performance metrics and confusion matrix plots.

---

## Future Improvements

- Implement real-time player tracking.
- Integrate with a video stream for live recognition.
- Enhance dataset with more player variations.

---

## Contributing

Feel free to fork this repository and submit pull requests for improvements!

---

## License

This project is licensed under the MIT License.

---

## Contact

For any inquiries, reach out at [your email or GitHub profile].

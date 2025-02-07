# Home work 4

### Folder Structure:

```
project_root/
│-- annotations/  # Stores XML annotations of train and test samples.
│-- images/       # Stores cropped images of ships and aircraft.
│-- models/       # Contains trained models.
│-- notebooks/    # Stores Jupyter notebooks for experiments.
│-- scripts/      # Python scripts for processing and training.
│-- logs/         # Logs from training and evaluation.
│-- results/      # Outputs such as plots and final predictions.
```
### Files:

- `scripts/xml_annotations_parser.py`: contains a class that crops and saves objects in the folders.
- `notebooks/small_objects_classification.ipynb`: trains the pretrained models below on our data and saves the best new models.
- Each `models/..._best.pth` file stores a trained model with the best accuracy.

### Models and Parameters:

| Model            | Parameters (Millions) | Size (MB) |
|------------------|-----------------------|-----------|
| ResNet18         | 11                    | ~45       |
| EfficientNet-B0  | 5.3                   | ~20       |
| RegNet_Y_400MF   | 9.2                   | ~40       |

This project focuses on detecting small objects like ships and aircraft using deep learning models. The training and evaluation pipeline includes ResNet18, EfficientNet-B0, and RegNet_Y_400MF, with results logged via Weights & Biases (wandb).

### Code Explanation:

- **Data Loading**: The `get_dataloader` function loads and preprocesses images into PyTorch DataLoaders.
- **Metric Calculation**: The `calculate_metrics` function computes accuracy, precision, recall, and F1 score.
- **Training Function**: The `train` function updates model weights using an optimizer and logs training loss.
- **Evaluation Function**: The `evaluate` function measures model performance on a validation set.
- **Training & Evaluation Pipeline**: The `train_and_evaluate` function automates training across multiple epochs, logs results using wandb, and saves the best-performing model.

The script supports three models: ResNet18, EfficientNet-B0, and RegNet_Y_400MF, ensuring a robust approach to small object detection in images.
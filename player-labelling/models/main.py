# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image


# Cleans up corrupted images
# Goes to the web scrapped images and checks if it can be opened and is convered to RGBA format for the models to train on
def clean_image_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    # Verifying that it is a valid image
                    img.verify()  

                    # Convert palette images with transparency to RGBA
                    if img.mode == 'P' and 'transparency' in img.info:
                        img = img.convert('RGBA')
                        img.save(file_path) 
            except Exception as e:
                os.remove(file_path)

# Define dataset path and clean
data_dir = '../images/nba_images/'
clean_image_folder(data_dir)

# Preprocessing and transformations
img_size = (224, 224)

# Augmented training dataset
    # Performs the following
train_transform = transforms.Compose([
    transforms.Resize((280, 280)),  # Even larger initial resize
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.RandomGrayscale(p=0.1)
])

# Resizing, normalizing and coverting to tensor format for validation data
val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Loading the dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Viewing the class labels
print("Class-to-index mapping:")
print(full_dataset.class_to_idx)

# Spitting into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply validation transform to val_dataset manually
val_dataset.dataset.transform = val_transform

# Medium sized batch size
batch_size = 16

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load ResNet50 with pre-trained weights
model = models.resnet50(pretrained=True)

# Unfreeze deeper layers for better feature learning
for param in model.parameters():
    param.requires_grad = False

for param in model.layer2.parameters():
    param.requires_grad = True

for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True


# Batch Normalization
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.requires_grad_(True)  
        module.train()  


# Modify the final classifier
num_classes = len(full_dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.7),  
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)


# Ensure that model parameters are set to require gradients correctly
for param in model.fc.parameters():
    param.requires_grad = True

# Try to use GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
class_counts = [len(os.listdir(os.path.join(data_dir, cls))) for cls in full_dataset.classes]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, 
                              label_smoothing=0.2)  

# Optimizer
optimizer = optim.AdamW([
    {'params': model.layer2.parameters(), 'lr': 5e-5},  # Reduced
    {'params': model.layer3.parameters(), 'lr': 8e-5},  # Reduced
    {'params': model.fc.parameters(), 'lr': 5e-4}  # Reduced
], weight_decay=1e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
epochs = 50
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_loss = float('inf')
patience = 5  # For early stopping
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"[Train] Predicted: {predicted.tolist()} | Actual: {labels.tolist()}")


    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)


    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        torch.save(model, 'full_model.pth')

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

    # Adjust learning rate
    scheduler.step()

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Accuracy and loss plots
plt.figure(figsize=(8, 6))
plt.plot(train_accuracies, label='Training Accuracy', color='royalblue')
plt.plot(val_accuracies, label='Validation Accuracy', color='darkorange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('training_vs_validation_accuracy.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss', color='crimson')
plt.plot(val_losses, label='Validation Loss', color='seagreen')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('training_vs_validation_loss.png')
plt.show()

## Confusion matrix
model.eval()
y_true, y_pred = [], []
y_pred_prob = []  # Store the predicted probabilities here
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  # Model outputs raw logits
        _, preds = torch.max(outputs, 1)  # Get class predictions
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
        # Apply softmax to outputs to get probabilities
        prob = torch.softmax(outputs, dim=1).cpu().numpy()  # Fix softmax applied on outputs, not y_pred
        y_pred_prob.extend(prob)

cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()  # Ensures labels aren't cut off

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')
plt.show()

# Precision-Recall Curve for each class
y_true_bin = label_binarize(y_true, classes=[i for i in range(num_classes)])  
y_pred_prob = torch.softmax(torch.tensor(y_pred_prob), dim=1).numpy()  # Make sure to pass correct tensor for softmax

plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
for i, color in zip(range(num_classes), colors):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
    plt.plot(recall, precision, label=f'Class {full_dataset.classes[i]}', color=color)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('precision_recall_curve.png')
plt.show()

# ROC Curve for each class
plt.figure(figsize=(8, 6))
for i, color in zip(range(num_classes), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {full_dataset.classes[i]} (AUC = {roc_auc:.2f})', color=color)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('roc_curve.png')
plt.show()

# Class-wise Accuracy
class_accuracies = []
for i in range(num_classes):
    class_indices = np.where(np.array(y_true) == i)[0]
    correct_preds = np.sum(np.array(y_pred)[class_indices] == i)
    accuracy = correct_preds / len(class_indices) * 100
    class_accuracies.append(accuracy)

plt.figure(figsize=(8, 6))
plt.bar(full_dataset.classes, class_accuracies, color='mediumseagreen', edgecolor='black')
plt.xlabel('Classes')
plt.ylabel('Accuracy (%)')
plt.title('Class-wise Accuracy')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('class_wise_accuracy.png')
plt.show()

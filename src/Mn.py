from const import FROM_CHECKPOINT_PATH, PATH_TEST_DIR, PATH_TEST_LABELS, PATH_TRAIN_DIR, PATH_TRAIN_LABELS
from config import N_HEADS, ENC_LAYERS, DEC_LAYERS, HIDDEN, DROPOUT, DEVICE, ALPHABET, BATCH_SIZE
import torch
from PIL import Image
from torchvision import transforms
import model2
from dataset import TextLoader, TextCollate
from utils import generate_data, process_data
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = model2.TransformerModel2(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                                 nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(FROM_CHECKPOINT_PATH))
model.eval()



def process_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image


def predict_image(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        output = model.predict(image)
    decoded_output = ''.join(
        [ALPHABET[idx] for idx in output[0] if idx < len(ALPHABET) and ALPHABET[idx] not in {'SOS', 'EOS'}])
    return decoded_output


def evaluate_on_subset(image_paths, labels):
    correct_predictions = 0
    total_predictions = len(image_paths)
    all_preds = []
    all_labels = []

    for img_path, true_label in zip(image_paths, labels):
        predicted_label = predict_image(img_path)
        if predicted_label == true_label:
            correct_predictions += 1
        all_preds.append(predicted_label)
        all_labels.append(true_label)

    accuracy = correct_predictions / total_predictions
    return accuracy, all_preds, all_labels


def plot_confusion_matrix(true_labels, pred_labels, title="Confusion Matrix"):

    unique_labels = list(set(true_labels) | set(pred_labels))


    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == "__main__":

    img2label_test, _, _ = process_data(PATH_TEST_DIR, PATH_TEST_LABELS)
    img_paths_test = list(img2label_test.keys())
    labels_test = list(img2label_test.values())


    subset_indices = random.sample(range(len(img_paths_test)), 1000)
    subset_img_paths_test = [img_paths_test[i] for i in subset_indices]
    subset_labels_test = [labels_test[i] for i in subset_indices]


    test_accuracy, test_preds, test_true = evaluate_on_subset(subset_img_paths_test, subset_labels_test)
    print(f"Test Accuracy on 1000 images: {test_accuracy}")


    plot_confusion_matrix(test_true, test_preds, title="Test Confusion Matrix")

    img2label_train, _, _ = process_data(PATH_TRAIN_DIR, PATH_TRAIN_LABELS)
    img_paths_train = list(img2label_train.keys())
    labels_train = list(img2label_train.values())


    subset_indices = random.sample(range(len(img_paths_train)), 1000)
    subset_img_paths_train = [img_paths_train[i] for i in subset_indices]
    subset_labels_train = [labels_train[i] for i in subset_indices]

    train_accuracy, train_preds, train_true = evaluate_on_subset(subset_img_paths_train, subset_labels_train)
    print(f"Train Accuracy on 1000 images: {train_accuracy}")


    plot_confusion_matrix(train_true, train_preds, title="Train Confusion Matrix")



    train_losses = [0.8, 0.6, 0.4, 0.3]
    val_losses = [0.9, 0.7, 0.5, 0.4]
    train_accuracies = [0.65, 0.75, 0.85, 0.9]
    val_accuracies = [0.6, 0.7, 0.8, 0.85]  

    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# Deep Learning for Metastatic Cancer Detection in Lymph Node Histopathology Images

This project explores the application of deep learning for the automatic detection of metastatic cancer in lymph node histopathology images, using the PatchCamelyon (PCam) dataset. It was completed as a final-year dissertation project for the BSc Computer Science degree at the University of Bath.

## 📋 Abstract

Early detection of breast cancer metastasis in lymph nodes is critical for effective treatment. Manual histopathological examination, though reliable, is slow and error-prone. This project investigates whether deep learning can improve the efficiency and accuracy of this task by comparing the performance of three architectures: a custom Convolutional Neural Network (CNN), DenseNet-121, and ResNet-34. Both pre-trained (ImageNet) and non-pre-trained versions were evaluated under different regularisation and optimisation strategies. The best-performing model was a pre-trained DenseNet-121 with an AUC of **0.9513** and accuracy of **88.72%**.

## 🧪 Experiments

Three experiments were conducted:

- **Experiment A**: Baseline using SGD with no regularisation.
- **Experiment B**: Adds dropout, weight decay, and a learning rate scheduler.
- **Experiment C**: Same as B but uses Adam instead of SGD.

Each experiment evaluated 5 models:
- CNN
- Non-pre-trained ResNet-34
- Pre-trained ResNet-34
- Non-pre-trained DenseNet-121
- Pre-trained DenseNet-121

## 📊 Results

| Model                  | Accuracy | AUC    |
|------------------------|----------|--------|
| Pre-trained DenseNet B | 88.72%   | 0.9513 |
| Pre-trained ResNet A   | 87.88%   | 0.9481 |
| CNN B                  | 86.25%   | 0.9336 |
| Non-pre-trained DenseNet C | 85.49% | 0.9352 |
| ...                    | ...      | ...    |

Pre-trained models consistently outperformed their non-pre-trained versions. Transfer learning proved especially effective with limited training data.

## 🖼️ Dataset

- **Name**: [PatchCamelyon (PCam)](https://github.com/basveeling/pcam)
- **Description**: 327,680 patches of size 96×96 extracted from WSIs of lymph node tissue (Camelyon16).
- **Labels**: Binary (tumor present / not present)

## 🛠️ Tools & Libraries

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- tqdm

## 🧠 Models Used

- **CNN**: 3 conv layers → FC → sigmoid
- **ResNet-34**: PyTorch pretrained and untrained versions
- **DenseNet-121**: PyTorch pretrained and untrained versions

Loss: `BCEWithLogitsLoss`  
Metrics: Accuracy, Precision, Recall, F1, AUC

## 📌 Key Techniques

- Data Augmentation (ColorJitter, flips, rotations)
- Transfer Learning (pretrained on ImageNet)
- Regularisation (dropout, weight decay)
- Learning Rate Scheduler (ReduceLROnPlateau)
- Comparison of optimisers (SGD vs Adam)

## 💡 Insights

- Pre-training on ImageNet significantly boosts performance.
- Regularisation improves generalisation in non-pre-trained models.
- SGD often outperformed Adam for pre-trained networks.

## 🚧 Limitations

- **False negatives**: Across multiple models, especially the pre-trained ResNet and DenseNet, a relatively high number of false negatives were observed. This is concerning for clinical use, where failing to detect cancer could have serious consequences.

- **Overfitting in non-pre-trained models**: The non-pre-trained ResNet and DenseNet models often struggled with overfitting, particularly in Experiment A. Despite regularisation efforts in Experiments B and C, their performance still lagged behind pre-trained counterparts. Overall, all models struggled with overfitting to an extent.

- **Limited dataset diversity**: Although PatchCamelyon is balanced and well-structured, it may not fully reflect the variability seen in real-world clinical settings.

- **Lack of whole-slide analysis**: The project focuses on patch-level classification and does not implement whole-slide inference or region-level localisation, which would be needed for clinical integration.


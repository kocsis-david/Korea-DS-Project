import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_results(image, target, prediction):
    from matplotlib.pyplot import imshow, show
    import numpy as np

    pred = np.transpose(prediction[0], (2, 1, 0))

    imshow(image)
    show()

    imshow(pred)
    show()

    imshow(target)
    show()

    image[1250:1750, :, :] = pred[1250:1270, :, :]*255
    imshow(image)
    show()

    image[1250:1750, :, :] = target[1250:1270, :, :]*255
    imshow(image)
    show()


def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def conf_matrix(val_dataset, predictions, plot=True):
    from sklearn.metrics import confusion_matrix
    num_classes = 9
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    i = 0
    for batch in predictions:
        for image in batch:
            pred = image.flatten()
            target = np.argmax(val_dataset[i][1], axis=0).flatten()

            conf_matrix_batch = confusion_matrix(pred, target, labels=range(num_classes))
            total_conf_matrix += conf_matrix_batch
            i += 1

    if plot:
        class_labels = list(range(num_classes))
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.heatmap(total_conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    return total_conf_matrix

def predict_windows(model, windows):
    predicted_windows = []

    for window in windows:
        input_tensor = torch.tensor(window.data, dtype=torch.float32)
        input_tensor = input_tensor.permute(2, 0, 1)  # Change shape from (256, 256, 4) to (4, 256, 256)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        prediction = model(input_tensor)
        predicted_windows.append(prediction[0].detach().numpy())

    return predicted_windows

def plotter(model_outputs, desired_outputs, input_images, onehot_to_rgb):
    num_samples = len(model_outputs)

    for i in range(num_samples):
        # Convert model output to RGB
        predicted_output = onehot_to_rgb(model_outputs[i])

        # Get input, desired output, and original image
        input_image = input_images[i]
        desired_output = desired_outputs[i]

        # Plot images
        plt.figure(figsize=(12, 4))
        
        # Original Input
        plt.subplot(1, 4, 1)
        plt.title('Original Input')
        plt.imshow(input_image)
        
        # Desired Output
        plt.subplot(1, 4, 2)
        plt.title('Desired Output')
        plt.imshow(desired_output)

        # Predicted Output
        plt.subplot(1, 4, 3)
        plt.title('Predicted Output')
        plt.imshow(predicted_output)

        # Input with Overlapping Prediction
        plt.subplot(1, 4, 4)
        plt.title('Input with Prediction')
        plt.imshow(input_image)
        plt.imshow(predicted_output, alpha=0.5)

        plt.tight_layout()
        plt.show()


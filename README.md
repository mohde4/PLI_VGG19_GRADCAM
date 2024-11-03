# PLI_VGG19_GRADCAM
Interpretability Analysis of Pixel-Level Interpretability and Grad-CAM on X-ray Imaging with VGG19
# PLI Model Implementation using TensorFlow/Keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt

# Load Pre-trained VGG19 Model + Higher Level Layers
base_model = VGG19(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv4').output)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def get_pli_heatmap(model, img_array, class_index, epsilon=1e-8):
    # Forward pass to get the predictions
    conv_output = model(img_array)
    
    # Pixel-level gradients for the class index
    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        preds = model(img_array)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)[0]

    # Normalize the gradients
    grads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads) + epsilon)

    # Average over all the filters to get the pixel-level importance
    pli_heatmap = np.mean(grads, axis=-1)
    pli_heatmap = np.maximum(pli_heatmap, 0)  # ReLU
    
    return pli_heatmap

def display_heatmap(heatmap, original_img_path):
    plt.matshow(heatmap)
    plt.colorbar()
    plt.title('PLI Heatmap')
    plt.show()

    img = image.load_img(original_img_path)
    img = image.img_to_array(img)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = np.concatenate([heatmap, heatmap, heatmap], axis=-1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    plt.imshow(overlay.astype('uint8'))
    plt.title('PLI Overlay on Original Image')
    plt.axis('off')
    plt.show()

# Example Usage
# Load and preprocess image
img_path = 'path_to_image.jpg'
img_array = load_and_preprocess_image(img_path)

# Get the PLI heatmap
class_index = 1  # Assuming binary classification (e.g., Pneumonia = 1, Normal = 0)
pli_heatmap = get_pli_heatmap(model, img_array, class_index)

# Display the heatmap
display_heatmap(pli_heatmap, img_path)

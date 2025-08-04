import os
import shutil
import random
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Set paths
SOURCE_DIR = r'E:\Codanics-SixMonthsAI-ML\Paddy Disease Classification\train_images' # Original dataset
OUTPUT_DIR = r'E:\Codanics-SixMonthsAI-ML\Paddy Disease Classification\balanced_images'     # Output balanced dataset
TARGET_SIZE = (224, 224)
TARGET_COUNT = 500

# Data augmentation generator
augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    output_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Copy up to 500 original images
    original_count = min(len(images), TARGET_COUNT)
    for i in range(original_count):
        src = os.path.join(class_path, images[i])
        dst = os.path.join(output_class_path, f'original_{i}.jpg')
        shutil.copy(src, dst)

    # If images < 500, generate more using augmentation
    if original_count < TARGET_COUNT:
        to_augment = images[:min(len(images), 50)]  # select 50 for augmentation
        needed = TARGET_COUNT - original_count
        print(f"Augmenting {needed} images for class '{class_name}'...")

        generated = 0
        for img_name in tqdm(to_augment):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, TARGET_SIZE)
            img = np.expand_dims(img, axis=0)

            for batch in augmentor.flow(img, batch_size=1):
                aug_img = batch[0].astype(np.uint8)
                aug_filename = f'augmented_{generated + original_count}.jpg'
                cv2.imwrite(os.path.join(output_class_path, aug_filename), aug_img)
                generated += 1
                if generated >= needed:
                    break
            if generated >= needed:
                break

print("\n Dataset balancing complete. Output saved in:", OUTPUT_DIR)

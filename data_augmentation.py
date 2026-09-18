import os
import numpy as np
import logging
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perform_data_augmentation(data_path, num_augmented_images_per_original):
    # Data Augmentation Configuration
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Iterate through all class subfolders
    class_folders = [
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    ]

    logger.info(f"Found classes: {class_folders}")

    for class_name in class_folders:
        class_path = os.path.join(data_path, class_name)
        image_files = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        logger.info(f"\nProcessing class: {class_name} ({len(image_files)} images)")

        augmented_count = 0

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)

            # Load image
            img = load_img(image_path)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Generate augmented images
            for j, batch in enumerate(train_datagen.flow(img_array, batch_size=1)):
                # Create filename with "_Augmented" suffix
                name, ext = os.path.splitext(image_file)
                augmented_filename = f"{name}_Augmented_{j}{ext}"
                augmented_path = os.path.join(class_path, augmented_filename)

                # Save augmented image (use PIL instead of plt.imsave)
                img_to_save = (batch[0]).astype(np.uint8)
                Image.fromarray(img_to_save).save(augmented_path)
                augmented_count += 1

                if j >= num_augmented_images_per_original - 1:
                    break

        logger.info(f"    {augmented_count} augmented images created")

    logger.info("\n  Data Augmentation completed!")


if __name__ == "__main__":
    # Execute data augmentation
    perform_data_augmentation("data/train", num_augmented_images_per_original=5)

from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm
import cv2


class AnnotationProcessor:
    """
    A class to process annotations from an XML file, crop images based on bounding boxes,
    and save the cropped images to specified folders.

    Attributes:
        train_folder (Path): Path to the folder containing training images.
        annotations_file (Path): Path to the XML file containing annotations.
        output_folder (Path): Path to the folder where cropped images will be saved.
        labels (List[str]): List of labels for which folders will be created.
    """

    def __init__(self, train_folder: str, annotations_file: str, output_folder: str) -> None:
        """
        Initialize the AnnotationProcessor.

        Args:
            train_folder (str): Path to the folder containing training images.
            annotations_file (str): Path to the XML file containing annotations.
            output_folder (str): Path to the folder where cropped images will be saved.
        """
        self.train_folder = Path(train_folder)
        self.annotations_file = Path(annotations_file)
        self.output_folder = Path(output_folder)
        self.labels: List[str] = ['ship', 'aircraft']  # Labels for creating folders

    def create_output_folders(self) -> None:
        """
        Create output folders for each label if they do not already exist.
        """
        for label in self.labels:
            (self.output_folder / label).mkdir(parents=True, exist_ok=True)

    def parse_annotations(self) -> BeautifulSoup:
        """
        Parse the XML file containing annotations.

        Returns:
            BeautifulSoup: A BeautifulSoup object representing the parsed XML.
        """
        with open(self.annotations_file, 'r') as f:
            return BeautifulSoup(f, 'xml')

    def process_image(self, image: BeautifulSoup) -> None:
        """
        Process a single image, extract bounding boxes, and save cropped images.

        Args:
            image (BeautifulSoup): An <image> element from the parsed XML.
        """
        image_name: str = image['name']
        image_path: Path = self.train_folder / image_name

        # Open the image
        img: Optional[cv2.Mat] = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Image {image_name} not found.")
            return

        # Process each bounding box
        for box in image.find_all('box'):
            label: str = box['label']
            xtl: int = int(float(box['xtl']))  # Bounding box coordinates
            ytl: int = int(float(box['ytl']))
            xbr: int = int(float(box['xbr']))
            ybr: int = int(float(box['ybr']))

            # Crop the image
            cropped_img: cv2.Mat = img[ytl:ybr, xtl:xbr]

            # Save the cropped image to the label folder
            output_path: Path = self.output_folder / label / f"{image_path.stem}_{xtl}_{ytl}_{xbr}_{ybr}.jpg"
            cv2.imwrite(str(output_path), cropped_img)

    def process_all_images(self) -> None:
        """
        Process all images from the annotations file, crop them, and save the results.
        """
        soup: BeautifulSoup = self.parse_annotations()
        self.create_output_folders()

        # Process each image with a progress bar
        for image in tqdm(soup.find_all('image'), desc="Processing images"):
            self.process_image(image)

        print("Processing completed!")


# Example usage
if __name__ == "__main__":
    # Paths to folders and files
    train_folder: str = 'Train'
    annotations_file: str = 'annotations/Train_annot.xml'
    output_folder: str = 'Saved images'

    # Create an instance of the class and start processing
    processor: AnnotationProcessor = AnnotationProcessor(train_folder, annotations_file, output_folder)
    processor.process_all_images()
    
# Home Task 4
### Folders:
- **annotations**: stores xml annotations of train and test samples
- **images**: stores cropped images of ships and aircrafts
### files:
- **xml_annotations_parser.py**: stores class that crop and save objects in the Folders
- **xml parser for CVAT images.ipynb**: stores all modifications, theory and attempt to understand XML parsing
- **small objects classification.ipynb** stores training of pretrained models below on our data and save best new models.
- each **..._best_loss.pth** stores new model and weights with lowest loss

| Model           | Parameters (Millions) | Size (MB) |
|-----------------|-----------------------|-----------|
| VGG19           | 143                   | ~500      |
| ResNet18        | 11                    | ~45       |
| EfficientNet-B0 | 5.3                   | ~20       |
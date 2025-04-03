import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
from pathlib import Path

class Predictor:
    def __init__(self, model, labelmap: Dict[int, str], visualize: bool = False, output_dir: Optional[Union[str, Path]] = None):
        self.model = model
        self.labelmap = labelmap
        self.visualize = visualize
        self.output_dir = Path(output_dir) if output_dir else None

        if self.visualize and not self.output_dir:
            raise ValueError("output_dir must be specified when visualize=True")
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_batch(self, batch_images: np.ndarray) -> Dict[str, np.ndarray]:
        probabilities = self.model.predict(batch_images, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)

        if self.visualize:
            self._visualize_batch(batch_images, predicted_classes, probabilities)

        return {
            'classes': predicted_classes,
            'probabilities': probabilities
        }

    def _visualize_batch(self, batch_images: np.ndarray, class_predictions: np.ndarray, probabilities: np.ndarray) -> None:
        
        batch_size = batch_images.shape[0]
        # plot 4 images per row
        nb_cols = 4
        nb_rows = batch_size//nb_cols + batch_size%nb_cols
        fig, axes = plt.subplots(nb_rows, nb_cols)
        
        for i in range(batch_size):
            image_row = i//nb_cols
            image_col = nb_cols % i
            axes[image_row, image_col].imshow(batch_images[i])
            axes[image_row, image_col].set_title(f"{self.labelmap[class_predictions[i]]} : {probabilities[i]}")
            
        if self.output_dir:
            plt.savefig(self.output_dir / f"pred_{i}.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @property
    def class_names(self) -> list:
        """Return ordered list of class names."""
        return [self.labelmap[i] for i in sorted(self.labelmap.keys())]
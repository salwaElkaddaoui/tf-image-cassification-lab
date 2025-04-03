from data import DataLoader
from matplotlib import pyplot as plt

if __name__=='__main__':
    data_loader = DataLoader(   img_size=256, 
                                batch_size=8, 
                                labelmap_path="/home/salwa/Documents/code/unet/data/pascalvoc_subset/labelmap_classification.json")
    
    dataset = data_loader.create_dataset(image_paths="/home/salwa/Documents/code/unet/data/pascalvoc_subset/train_image_paths.txt", training=True)

    for (images, labels) in dataset:
        print(labels.numpy())
        print(images.shape)
        print("\n")

        plt.imshow(images[0, ...].numpy())
        plt.show()
        break
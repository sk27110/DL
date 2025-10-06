from src.datasets.load_aircraft_data import load_aircraft_dataset
from src.datasets.aircraft_dataset import AircraftDataset
from src.transforms.transforms import train_transform, resize_crop_transform, color_jitter_transform, rotation_transform, graysacle_transform, perspective_trandform
from src.utils.denormalize import denormalize
import matplotlib.pyplot as plt

DATAPATH="data/aircraft_data"
num_folders=1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



if __name__=="__main__":
    # load_aircraft_dataset(DATAPATH)
    dataset = AircraftDataset(DATAPATH, num_folders)
    tensor_image = dataset.__getitem__(0, resize_crop_transform)['data_object']
    denormalize_img = denormalize(tensor_image, mean, std)
    img_np = denormalize_img.permute(1, 2, 0).clamp(0, 1)  # clamp ограничивает диапазон [0,1]
    # Отображаем
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()
    
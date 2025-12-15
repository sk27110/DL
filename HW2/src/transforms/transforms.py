from torchvision import transforms
import ast


class Transforms_v1:
    def __init__(
        self,
        mean, 
        std,
        resize
    ):
        self.mean = mean
        self.std = std
        self.resize = ast.literal_eval(resize)

        self.test = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ast.literal_eval(self.mean),
                std=ast.literal_eval(self.std)
            )
        ])
        self.train = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.RandomRotation(20), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),

            transforms.ColorJitter( 
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02
            ), 
            transforms.RandomGrayscale(p=0.05), 


            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ast.literal_eval(self.mean),
                std=ast.literal_eval(self.std)
            )
        ])



class Transforms_v2:
    def __init__(
        self,
        mean, 
        std,
        resize
    ):
        self.mean = mean
        self.std = std
        self.resize = ast.literal_eval(resize)

        self.test = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ast.literal_eval(self.mean),
                std=ast.literal_eval(self.std)
            )
        ])
        
        self.train = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.2  
            ),
            
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,           
                    translate=(0.1, 0.1), 
                    scale=(0.8, 1.2),  
                    shear=10
                )
            ], p=0.3),

            transforms.ColorJitter( 
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02
            ), 
            transforms.RandomGrayscale(p=0.05), 
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5) 
            ], p=0.2),
            
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
            ], p=0.3),
            
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ast.literal_eval(self.mean),
                std=ast.literal_eval(self.std)
            ),
            
            transforms.RandomErasing(
                p=0.3,                   
                scale=(0.02, 0.2),     
                ratio=(0.3, 3.3),       
                value='random'      
            )
        ])
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
import torchvision.transforms as T
import torch
# https://docs.ultralytics.com/tasks/classify/#how-do-i-train-a-yolo11-model-for-image-classification
"""
class CustomizedDataset(ClassificationDataset):
    # A customized dataset class for image classification with enhanced data augmentation transforms.

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        # Initialize a customized classification dataset with enhanced data augmentation transforms.
        super().__init__(root, args, augment, prefix)

        # Add your custom training transforms here
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        )

        # Add your custom validation transforms here
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms

class CustomizedTrainer(ClassificationTrainer):
    # A customized trainer class for YOLO classification models with enhanced dataset handling.

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        # Build a customized dataset for classification training and the validation during training.
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

class CustomizedValidator(ClassificationValidator):
    # A customized validator class for YOLO classification models with enhanced dataset handling.

    def build_dataset(self, img_path: str, mode: str = "train"):
        # Build a customized dataset for classification standalone validation.
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)
    
model = YOLO("yolo11n-cls.pt")
model.train(data="imagenet1000", trainer=CustomizedTrainer, epochs=10, imgsz=224, batch=64)
model.val(data="imagenet1000", validator=CustomizedValidator, imgsz=224, batch=64)

"""

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model YOLO11n-cls
results = model.train(data="datasets/split_data", epochs=100, imgsz=640)

# Print specific metrics
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("Class indices for average precision:", results.box.ap_class_index)
print("Class-specific results:", results.box.class_result)
print("F1 score:", results.box.f1)
print("F1 score curve:", results.box.f1_curve)
print("Overall fitness score:", results.box.fitness)
print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean average precision for different IoU thresholds:", results.box.maps)
print("Mean results for different metrics:", results.box.mean_results)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Precision curve:", results.box.p_curve)
print("Precision values:", results.box.prec_values)
print("Specific precision metrics:", results.box.px)
print("Recall:", results.box.r)
print("Recall curve:", results.box.r_curve)

model.val(data="datasets/split_data/test", imgsz=224, batch=64)

# Export the model
model.export(format="onnx")
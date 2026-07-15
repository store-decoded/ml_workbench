from __future__ import print_function
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from datasets import load_from_disk,load_dataset,DatasetDict,Dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor, DefaultDataCollator, TrainingArguments, Trainer,AutoConfig
from torchvision.transforms import Compose, ColorJitter, ToTensor,Grayscale,Normalize,Resize,Lambda
import evaluate
import torch
import torch.nn as nn
from linformer import Linformer
from vit_pytorch.efficient import ViT
# import accuracy

accuracy = evaluate.load("accuracy")

def get_vit_efficient_model(checkpoint:str,dataset_dict:DatasetDict):

    num_of_classes = dataset_dict['train'].features['label'].num_classes
    efficient_transformer = Linformer(dim=128, seq_len=49+1, depth=12, heads=8, k=64)
    model = ViT(image_size=224, patch_size=32, num_classes=num_of_classes, dim=128 ,transformer=efficient_transformer)

    return model



def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)



def get_vit_auto_model(checkpoint:str,dataset_dict:DatasetDict):
    num_of_classes = dataset_dict['train'].features['label'].num_classes
    labels = dataset_dict["train"].features["label"].names
    print(f"num of classes {num_of_classes}")
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,

        num_labels=len(labels),

        id2label=id2label,

        label2id=label2id,

    )
    return model

def get_transformed_dataset(path:str,image_processor:AutoImageProcessor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    jitter = Compose(
        [

            ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.5),
            Grayscale(num_output_channels=1),
            Resize(size=(224, 224), antialias=True),
            ToTensor(),
            Lambda(lambda x: x.repeat(3,1,1)),
            normalize
        ]
    )
    def transforms(examples):
        examples["pixel_values"] = [jitter(image) for image in examples["image"]]
        del examples["image"]
        return examples


    dset = load_dataset(path,cache_dir='datasets/cache_dir')
    # dset = load_dataset('imagefolder',data_dir='datasets/split_data')
    dataset_dict = DatasetDict({"test":dset['test'],'train':dset['train'],'validation':dset['validation']})
    transformed_dataset = dataset_dict.with_transform(transforms)
    return transformed_dataset


def start_train():
    print(f"cuda availabel => {torch.cuda.is_available()}")
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint,use_fast=True)
    dataset_dict = get_transformed_dataset('AliAsh/glyphs_classifier_data',image_processor)

    model = get_vit_auto_model(checkpoint,dataset_dict)
    # model = get_vit_efficient_model(checkpoint,dataset_dict)
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(

        output_dir="models/vit_final",

        remove_unused_columns=False,

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=5e-5,

        per_device_train_batch_size=24,

        gradient_accumulation_steps=4,

        per_device_eval_batch_size=24,

        num_train_epochs=5,

        warmup_ratio=0.1,

        logging_steps=10,

        load_best_model_at_end=True,

        metric_for_best_model="accuracy",
        
        use_cpu=False

    )

    trainer = Trainer(

        model=model,

        args=training_args,

        data_collator=data_collator,

        train_dataset=dataset_dict["train"],

        eval_dataset=dataset_dict["test"],

        processing_class=image_processor,

        compute_metrics=compute_metrics,

        
    )
    trainer.train()

def classify_image(checkpoint:str,samples_path:str,expection:str):
    uncode_mappings = pd.read_csv('glyph_unicode_mappings.csv')
    config = AutoConfig.from_pretrained(checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)
    images = glob.glob(f'{samples_path}/*.png')
    font_unmapped_detected = {}
    score = 0
    for img in tqdm(images):
        image = Image.open(img)
        inputs = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_indices = torch.argmax(logits, dim=1)  
        class_labels = model.config.id2label  
        predicted_labels = [class_labels[idx.item()] for idx in predicted_indices]
        resolved_character = uncode_mappings.loc[uncode_mappings['char_name'].eq(predicted_labels[0])]['char_value'].values[0]
        glyph_name = img.split('/')[-1].replace('.png','')
        # font_unmapped_detected[f"{glyph_name:#06x}".replace("0x", "")] = "0x" + hex(ord(resolved_character)).replace("0x", "").zfill(4)
        # font_unmapped_detected[glyph_name] = resolved_character
        if predicted_labels[0] == expection:
            # print('percision',score/len(images))
            score += 1
        # print(f"predicted class {predicted_labels} on file {img} ({resolved_character})")
    return score/len(images)

def evaluate_model(checkpoint:str, dataset_dict:DatasetDict, device='cpu'):
    test_dataset = dataset_dict['test']
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, config=config, ignore_mismatched_sizes=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    model.eval() 
    correct_predictions = 0
    total_predictions = 0

    model.to(device)
    
    image_processor = AutoImageProcessor.from_pretrained(model.name_or_path)

    with torch.no_grad():
        for batch in tqdm(test_dataset):
            images = batch['image'] 
            label = batch['label']  

            inputs = image_processor(images=images, return_tensors="pt").to(device)

            label_tensor = torch.tensor(label, dtype=torch.long).to(device)

            logits = model(**inputs).logits 
            predicted_indices = torch.argmax(logits, dim=1) 

            correct_predictions += (predicted_indices == label_tensor).sum().item()
            total_predictions += 1  

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy


#https://huggingface.co/docs/transformers/v4.56.2/en/tasks/image_classification#image-classification 
if __name__ == "__main__":
    # predicted_labels = classify_image("automodel/checkpoint-195",
    #                './classifier_dset/test/ARABIC_LETTER_JEEM/0006b280-7a16-11ee-9a14-02420a0013f2_Microsoft Sans Serif,Bold_862_uni062C.png')
    checkpoint = "automodel/checkpoint-800"
    start_train()

    dset = load_dataset('AliAsh/glyphs_classifier_data',cache_dir='datasets/cache_dir')
    # va = get_transformed_dataset("DFsfdsa")
    # print(len(va['test']))
    # print(len(va['train']))
    # for s in va['test']:

    # dset_path = 'datasets/dataset_cache_folder'
    # dataset_dict = load_dataset(dset_path,data_files={'train':'train/data-00000-of-00001.arrow',
    #                                                     'test':'test/data-00000-of-00001.arrow',
    #                                                     'validation':'validation/data-00000-of-00001.arrow'})
    # print(dataset_dict['test'])
    # dataset_dict = get_transformed_dataset('datasets/dataset_cache_folder')
    # accuracy = evaluate_model(checkpoint,dataset_dict,'cuda')
    # print(accuracy)
    # start_train()
    # all_chars = glob.glob('/home/v0id/code/playground/play/datasets/split_data/test/*')
    # for char in all_chars:
    #     expection = Path(char).name
    #     print("START",expection)
    #     result = classify_image(checkpoint,char,expection)
    #     print(f"got {result} correct on {expection}")

    # glyph_name = 15455
    # val = f"{glyph_name:#06x}".replace("0x", "")
    # print(val)


    # dataset_dict = dataset_dict.with_transform(transforms)


    # jitter = Compose(
    #     [

    #         ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.5),
    #         Grayscale(num_output_channels=1),
    #         Resize(size=(224, 224), antialias=True),
    #         ToTensor(),
    #         Lambda(lambda x: x.repeat(3,1,1))
    #     ]
    # )
    # def transforms(examples):
    #     examples["pixel_values"] = [jitter(image) for image in examples["image"]]
    #     del examples["image"]
    #     return examples

    # # dataset_dict = load_dataset(path,data_files={'train':'train/data-00000-of-00001.arrow',
    # #                                                                           'test':'test/data-00000-of-00001.arrow',
    # #                                                                           'validation':'validation/data-00000-of-00001.arrow'})
    # import datasets
    # test_split = load_dataset('imagefolder',data_dir='datasets/split_data/test/')
    # train_split = load_dataset('imagefolder',data_dir='datasets/split_data/train/')
    # val_split = load_dataset('imagefolder',data_dir='datasets/split_data/validation/')
    # dataset_dict = DatasetDict({"test":test_split,'train':train_split,'validation':val_split})
    
    # dataset_dict = dataset_dict.with_transform(transforms)



    # {'eval_loss': 0.7671417593955994, 'eval_accuracy': 0.9697104677060133, 'eval_runtime': 108.035, 'eval_samples_per_second': 20.78, 'eval_steps_per_second': 0.87, 'epoch': 5.0}
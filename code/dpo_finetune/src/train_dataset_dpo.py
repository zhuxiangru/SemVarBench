import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import os
import io
import json
import math
from torchvision.transforms.functional import crop
import random


# import base64
# # base64输出的是string, 不是byte
# def image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         image_data = image_file.read()
#         base64_data = base64.b64encode(image_data).decode('utf-8')
#     return base64_data

# 输出的是byte
def image_to_bytes(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    return image_data


class DiffusersDPODataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        prompt_root,
        resolution,
        random_crop,
        no_hflip,
        tokenizer_one,
        tokenizer_two,
        reward_match_root=None,
        reward_mismatch_root=None,
        high_threshold=1.0,
        zero_threshold=0,
        threshold_filter="all",
        # dataset_root=None,
    ):

        self.resolution = resolution
        self.random_crop = random_crop
        self.no_hflip = no_hflip
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            print (self.instance_data_root)
            raise ValueError("Instance images root doesn't exists.")


        #read image
        instance_images_id2path = load_images(instance_data_root)
        instance_ids = list(instance_images_id2path.keys())
        instance_pure_ids = list(set(["_".join(id.split("_")[:-1]) for id in instance_ids]))
        # instance_images_path = list(instance_images_id2path.values())

        #read prompt
        instance_prompt_id2prompts = load_prompts(prompt_root)
        # instance_prompt = [instance_prompt_id2prompts[id] for id in instance_ids]

        #read reward
        instance_match_id2scores = load_match_scores(reward_match_root)
        instance_mismatch_id2scores = load_mismatch_scores(reward_mismatch_root)
        # print (instance_match_id2scores)

        # match_scores = torch.zeros((len(instance_ids), 1))
        # mismatch_scores = torch.zeros((len(instance_ids), 1))

        instance_final_id = []
        instance_final_prompt = []
        instance_final_negative_prompt = []
        instance_final_image_path = []
        instance_final_negative_image_path = []
        instance_final_match_scores = []
        instance_final_mismatch_scores = []

        # read training set which satisfy the reward
        for id in instance_pure_ids:
            id_a = id + "_a"
            id_b = id + "_b"
            id_c = id + "_c"
            s11 = instance_match_id2scores[id_a]
            s22 = instance_match_id2scores[id_b]
            s33 = instance_match_id2scores[id_c]
            s12 = instance_mismatch_id2scores[id_a]
            s21 = instance_mismatch_id2scores[id_b]
            
            if threshold_filter == "all":
                if s11 > high_threshold and s22 > high_threshold and s33 > high_threshold and \
                    s11 - s12 > 0 and s11 - s21 > 0 and s22 - s12 > 0 and s22 - s21 > 0 and \
                    math.fabs((s11 - s12) - (s22 - s21)) < zero_threshold and math.fabs((s11-s21) - (s22-s12)) < zero_threshold:

                    instance_final_id.append(id_a)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_a])
                    instance_final_negative_image_path.append(instance_images_id2path[id_b])
                    # instance_final_match_scores.append(instance_match_id2scores[id_a])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_a])
                    
                    instance_final_id.append(id_b)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_image_path.append(instance_images_id2path[id_b])
                    instance_final_negative_image_path.append(instance_images_id2path[id_a])
                    # instance_final_match_scores.append(instance_match_id2scores[id_b])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])
                    
                    instance_final_id.append(id_c)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_c])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_c])
                    instance_final_negative_image_path.append(instance_images_id2path[id_b])
                    # instance_final_match_scores.append(instance_match_id2scores[id_c])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_c])
                    
                    instance_final_id.append(id_b)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_c])
                    instance_final_image_path.append(instance_images_id2path[id_b])
                    instance_final_negative_image_path.append(instance_images_id2path[id_c])
                    # instance_final_match_scores.append(instance_match_id2scores[id_b])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])
                    
                    # print (instance_images_id2path[id_a])
                else:
                    continue
            elif threshold_filter == "part":
                if s11 > high_threshold and s22 > high_threshold:
                    instance_final_id.append(id_a)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_a])
                    instance_final_negative_image_path.append(instance_images_id2path[id_b])
                    # instance_final_match_scores.append(instance_match_id2scores[id_a])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_a])
                    
                    instance_final_id.append(id_b)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_image_path.append(instance_images_id2path[id_b])
                    instance_final_negative_image_path.append(instance_images_id2path[id_a])
                    # instance_final_match_scores.append(instance_match_id2scores[id_b])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])
                    
                    instance_final_id.append(id_c)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_c])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_c])
                    instance_final_negative_image_path.append(instance_images_id2path[id_b])
                    # instance_final_match_scores.append(instance_match_id2scores[id_c])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_c])
                    
                    instance_final_id.append(id_b)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_c])
                    instance_final_image_path.append(instance_images_id2path[id_b])
                    instance_final_negative_image_path.append(instance_images_id2path[id_c])
                    # instance_final_match_scores.append(instance_match_id2scores[id_b])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])
                    
                else:
                    continue
            else:
                instance_final_id.append(id_a)
                instance_final_prompt.append(instance_prompt_id2prompts[id_a])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_image_path.append(instance_images_id2path[id_a])
                instance_final_negative_image_path.append(instance_images_id2path[id_b])
                # instance_final_match_scores.append(instance_match_id2scores[id_a])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_a])
                
                instance_final_id.append(id_b)
                instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_a])
                instance_final_image_path.append(instance_images_id2path[id_b])
                instance_final_negative_image_path.append(instance_images_id2path[id_a])
                # instance_final_match_scores.append(instance_match_id2scores[id_b])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])

                instance_final_id.append(id_c)
                instance_final_prompt.append(instance_prompt_id2prompts[id_c])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_image_path.append(instance_images_id2path[id_c])
                instance_final_negative_image_path.append(instance_images_id2path[id_b])
                # instance_final_match_scores.append(instance_match_id2scores[id_c])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_c])

                instance_final_id.append(id_b)
                instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_c])
                instance_final_image_path.append(instance_images_id2path[id_b])
                instance_final_negative_image_path.append(instance_images_id2path[id_c])
                # instance_final_match_scores.append(instance_match_id2scores[id_b])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])

        self.instance_id = instance_final_id
        self.instance_prompt = instance_final_prompt
        self.instance_negative_prompt = instance_final_negative_prompt
        self.instance_image_path = instance_final_image_path
        self.instance_negative_image_path = instance_final_negative_image_path
        self.instance_match_scores = instance_final_match_scores
        self.instance_mismatch_scores = instance_final_mismatch_scores

        # Preprocessing the datasets.
        self.train_resize = transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.RandomCrop(self.resolution) if self.random_crop else transforms.CenterCrop(self.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])
    
        # self.num_instance_images = len(self.instance_prompt)
        # self._length = self.num_instance_images
        self._length = len(self.instance_id)

        self.features = ["jpg_0", "jpg_1", "caption", "label_0"]
        self.num_rows = self._length


    def __len__(self):
        return self._length
    
    def __repr__(self):
        return f"Dataset({{\n    features: {self.features},\n    num_rows: {self.num_rows}\n}})"

    def __getitem__(self, index):
        example = {}
        
        image_id = self.instance_id[index]
        image_chosen_path = self.instance_image_path[index]
        image_reject_path = self.instance_negative_image_path[index]
        
        example["jpg_0"] = image_to_bytes(image_chosen_path)
        example["jpg_1"] = image_to_bytes(image_reject_path)
        example["label_0"] = 1 # chosen的图片位于第一个位置,reject的图片位于第二个位置，
        example["caption"] = self.instance_prompt[index]
        
        example = self.preprocess_train(example)

        return example
    
    def preprocess_train(self, example):
        all_pixel_values = []
        image = Image.open(io.BytesIO(example["jpg_0"])).convert("RGB")
        original_size = (image.height, image.width)
        crop_top_lefts = []

        for col_name in ["jpg_0", "jpg_1"]:
            image = Image.open(io.BytesIO(example[col_name])).convert("RGB")
            if col_name == "jpg_1":
                # Need to bring down the image to the same resolution.
                # This seems like the simplest reasonable approach.
                # "::-1" because PIL resize takes (width, height).
                # print ("original jpg_1=", image.size)
                image = image.resize(original_size[::-1])
                # print ("jpg_1=", image.size)
            # else:
                # print ("jgp_0=", image.size)
            pixel_values = self.to_tensor(image)
            all_pixel_values.append(pixel_values)

        im_tup = all_pixel_values
        label_0 = example["label_0"]

        if label_0 == 0:
            im_tup = im_tup[::-1]
        combined_im = torch.cat(im_tup, dim=0)  # no batch dim
        # Resize.
        combined_im = self.train_resize(combined_im)
        # Flipping.
        if not self.no_hflip and random.random() < 0.5:
            combined_im = self.train_flip(combined_im)
        # Cropping.
        if not self.random_crop:
            y1 = max(0, int(round((combined_im.shape[1] - self.resolution) / 2.0)))
            x1 = max(0, int(round((combined_im.shape[2] - self.resolution) / 2.0)))
            combined_im = self.train_crop(combined_im)
        else:
            y1, x1, h, w = self.train_crop.get_params(combined_im, (self.resolution, self.resolution))
            combined_im = self.crop(combined_im, y1, x1, h, w)
        crop_top_left = (y1, x1)
        combined_im = self.normalize(combined_im)

        example["pixel_values"] = combined_im
        example["original_sizes"] = original_size
        example["crop_top_lefts"] = crop_top_left
        tokens_one, tokens_two = tokenize_caption([self.tokenizer_one, self.tokenizer_two], example)
        example["input_ids_one"] = tokens_one
        example["input_ids_two"] = tokens_two
        return example


def tokenize_caption(tokenizers, example):
    # captions = []
    # for caption in examples["caption"]:
    #     captions.append(caption)
    caption = example["caption"]

    tokens_one = tokenizers[0](
        caption, truncation=True, padding="max_length", max_length=tokenizers[0].model_max_length, return_tensors="pt"
    ).input_ids
    tokens_two = tokenizers[1](
        caption, truncation=True, padding="max_length", max_length=tokenizers[1].model_max_length, return_tensors="pt"
    ).input_ids

    return tokens_one, tokens_two

def load_images(instance_data_root):
    # instance_images_path = os.listdir(instance_data_root)
    instance_images_id2path = {}
    for root, dir, files in os.walk(instance_data_root):
        if dir != []:
            continue

        for file in files:
            if file.split(".")[-1] in ["jpg", "png"]:
                path = os.path.join(root, file)
                id = "_".join(path.split("/")[-1].split(".")[0].split("_")[:-1])
                if id != "":
                    instance_images_id2path.update({id: path})

    # print (instance_images_id2path)
    # instance_prompt = os.listdir(instance_data_root)    #read prompt from file name
    # instance_prompt.sort(key=lambda x: int(x.split("_")[1].split('.')[0]))  #sort
    return instance_images_id2path

def load_prompts(prompt_root):
    instance_prompt_id2prompts = {}
    for root, dir, files in os.walk(prompt_root):
        if dir != []:
            continue

        for file in files:
            if file.split(".")[-1] in ["txt"]:
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        id, prompt_a, prompt_b, prompt_c = line.strip().split("\t")
                        id_a = id + "_a"
                        id_b = id + "_b"
                        id_c = id + "_c"
                        instance_prompt_id2prompts.update({id_a: prompt_a, id_b: prompt_b, id_c: prompt_c})

    # print (instance_prompt_id2prompts)
    return instance_prompt_id2prompts


def load_match_scores(reward_match_root):
    instance_match_id2scoress = {}
    for root, dir, files in os.walk(reward_match_root):
        if dir != []:
            continue

        for file in files:
            if file.split(".")[-1] in ["txt"]:
                path = os.path.join(root, file)
                # if "question" not in path and "program" not in path and "descriptions" not in path:
                if "score" in path:
                    # print (path)
                    with open(path, "r", encoding="utf-8") as infile:
                        for line in infile:
                            r = json.loads(line.strip())
                            try:
                                id_a = r["caption_0"][0]["prompt_id"]
                                id_b = r["caption_1"][0]["prompt_id"]
                                id_c = r["caption_2"][0]["prompt_id"]
                                score_a = r["caption_0"][0]["score"]
                                score_b = r["caption_1"][0]["score"]
                                score_c = r["caption_2"][0]["score"]
                                instance_match_id2scoress.update({id_a: score_a, id_b: score_b, id_c: score_c})
                            except:
                                print (r["caption_0"])
                                print (list(r["caption_0"]))
                                # print (r["caption_0"][0])
                                print (path)

    # print (instance_match_id2scoress)
    return instance_match_id2scoress

def load_mismatch_scores(reward_mismatch_root):
    instance_mismatch_id2scoress = {}
    for root, dir, files in os.walk(reward_mismatch_root):
        if dir != []:
            continue

        for file in files:
            if file.split(".")[-1] in ["txt"]:
                path = os.path.join(root, file)
                # if "question" not in path and "program" not in path and "descriptions" not in path:
                if "score" in path:
                    # print (path)
                    with open(path, "r", encoding="utf-8") as infile:
                        for line in infile:
                            r = json.loads(line.strip())
                            try:
                                id_a = r["caption_0"][0]["prompt_id"]
                                id_b = r["caption_1"][0]["prompt_id"]
                                score_a = r["caption_0"][0]["score"]
                                score_b = r["caption_1"][0]["score"]
                                instance_mismatch_id2scoress.update({id_a: score_a, id_b: score_b})
                            except:
                                print (r["caption_0"])
                                print (list(r["caption_0"]))
                                # print (r["caption_0"][0])
                                print (path)
    return instance_mismatch_id2scoress
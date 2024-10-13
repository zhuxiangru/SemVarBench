import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import os
import json
import math
from torchvision.transforms.functional import crop
import random

class Equivariance_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        prompt_root,
        tokenizer_one,
        tokenizer_two,
        resolution=1024,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
        reward_match_root=None,
        reward_mismatch_root=None,
        high_threshold=1.0,
        zero_threshold=0,
        threshold_filter="all",
        # dataset_root=None,
    ):
        self.resolution = resolution
        self.center_crop = center_crop
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        print (instance_data_root)
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

        # match_scores = torch.zeros((len(instance_ids), 1))
        # mismatch_scores = torch.zeros((len(instance_ids), 1))

        instance_final_id = []
        instance_final_prompt = []
        instance_final_negative_prompt = []
        instance_final_image_path = []
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
                    # instance_final_match_scores.append(instance_match_id2scores[id_a])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_a])
                    print (instance_images_id2path[id_a])

                    instance_final_id.append(id_b)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_image_path.append(instance_images_id2path[id_b])
                    # instance_final_match_scores.append(instance_match_id2scores[id_b])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])
                    print (instance_images_id2path[id_b])

                    instance_final_id.append(id_c)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_c])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_c])
                    # instance_final_match_scores.append(instance_match_id2scores[id_c])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_c])
                    print (instance_images_id2path[id_c])
                else:
                    continue
            elif threshold_filter == "part":
                if s11 > high_threshold and s22 > high_threshold:
                    instance_final_id.append(id_a)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_a])
                    # instance_final_match_scores.append(instance_match_id2scores[id_a])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_a])

                    instance_final_id.append(id_b)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_a])
                    instance_final_image_path.append(instance_images_id2path[id_b])
                    # instance_final_match_scores.append(instance_match_id2scores[id_b])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])

                    instance_final_id.append(id_c)
                    instance_final_prompt.append(instance_prompt_id2prompts[id_c])
                    instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                    instance_final_image_path.append(instance_images_id2path[id_c])
                    # instance_final_match_scores.append(instance_match_id2scores[id_c])
                    # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_c])
                else:
                    continue
            else:
                instance_final_id.append(id_a)
                instance_final_prompt.append(instance_prompt_id2prompts[id_a])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_image_path.append(instance_images_id2path[id_a])
                # instance_final_match_scores.append(instance_match_id2scores[id_a])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_a])

                instance_final_id.append(id_b)
                instance_final_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_a])
                instance_final_image_path.append(instance_images_id2path[id_b])
                # instance_final_match_scores.append(instance_match_id2scores[id_b])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_b])

                instance_final_id.append(id_c)
                instance_final_prompt.append(instance_prompt_id2prompts[id_c])
                instance_final_negative_prompt.append(instance_prompt_id2prompts[id_b])
                instance_final_image_path.append(instance_images_id2path[id_c])
                # instance_final_match_scores.append(instance_match_id2scores[id_c])
                # instance_final_mismatch_scores.append(instance_mismatch_id2scores[id_c])

        self.instance_id = instance_final_id
        self.instance_prompt = instance_final_prompt
        self.instance_negative_prompt = instance_final_negative_prompt
        self.instance_image_path = instance_final_image_path
        self.instance_match_scores = instance_final_match_scores
        self.instance_mismatch_scores = instance_final_mismatch_scores

        # self.num_instance_images = len(self.instance_prompt)
        # self._length = self.num_instance_images
        self._length = len(self.instance_id)

        # img_transforms = []

        # if resize:
        #     img_transforms.append(
        #         transforms.Resize(
        #             size, interpolation=transforms.InterpolationMode.BILINEAR
        #         )
        #     )
        # if center_crop:
        #     img_transforms.append(transforms.CenterCrop(size))
        # if color_jitter:
        #     img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        # if h_flip:
        #     img_transforms.append(transforms.RandomHorizontalFlip())

        # self.image_transforms = transforms.Compose(
        #     [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        # )

        # Preprocessing the datasets.
        self.h_flip = h_flip
        self.center_crop = center_crop
        self.train_resize = transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(self.resolution) if center_crop else transforms.RandomCrop(self.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # image
        instance_image = Image.open(self.instance_image_path[index])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        original_size = (instance_image.height, instance_image.width)
        instance_image = self.train_resize(instance_image)
        if self.h_flip and random.random() < 0.5:
            # flip
            instance_image = self.train_flip(instance_image)
        if self.center_crop:
            y1 = max(0, int(round((instance_image.height - self.resolution) / 2.0)))
            x1 = max(0, int(round((instance_image.width - self.resolution) / 2.0)))
            instance_image = self.train_crop(instance_image)
        else:
            y1, x1, h, w = self.train_crop.get_params(instance_image, (self.resolution, self.resolution))
            instance_image = crop(instance_image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        instance_image = self.train_transforms(instance_image)

        example["original_sizes"] = original_size
        example["crop_top_lefts"] = crop_top_left
        example["pixel_values"] = instance_image

        # prompt
        prompt = self.instance_prompt[index]
        # negative_prompt = self.instance_negative_prompt[index]
        tokens_one = tokenize_prompt(self.tokenizer_one, prompt)
        tokens_two = tokenize_prompt(self.tokenizer_two, prompt)
        # negative_token_one = tokenize_prompt(self.tokenizer_one, negative_prompt)
        # negative_token_two = tokenize_prompt(self.tokenizer_two, negative_prompt)

        example["input_ids_one"] = tokens_one
        example["input_ids_two"] = tokens_two
        # example["input_ids_neg_one"] = negative_token_one
        # example["input_ids_neg_two"] = negative_token_two

        # match score & mismatch score
        # example["match_score"] = self.instance_match_scores[index]
        # example["mismatch_score"] = self.instance_mismatch_scores[index]

        # sdv2-1 version
        # example["instance_images"] = self.image_transforms(instance_image)
        # example["instance_prompt_ids"] = self.tokenizer(
        #     self.instance_prompt[index],
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.tokenizer.model_max_length,
        # ).input_ids
        # example["reward"] = self.reward[index]

        # sd xl original version
        # images = [image.convert("RGB") for image in examples[image_column]]
        # # image aug
        # original_sizes = []
        # all_images = []
        # crop_top_lefts = []
        # for image in images:
        #     original_sizes.append((image.height, image.width))
        #     image = train_resize(image)
        #     if args.random_flip and random.random() < 0.5:
        #         # flip
        #         image = train_flip(image)
        #     if args.center_crop:
        #         y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
        #         x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
        #         image = train_crop(image)
        #     else:
        #         y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
        #         image = crop(image, y1, x1, h, w)
        #     crop_top_left = (y1, x1)
        #     crop_top_lefts.append(crop_top_left)
        #     image = train_transforms(image)
        #     all_images.append(image)

        # examples["original_sizes"] = original_sizes
        # examples["crop_top_lefts"] = crop_top_lefts
        # examples["pixel_values"] = all_images
        # tokens_one, tokens_two = tokenize_captions(examples)
        # examples["input_ids_one"] = tokens_one
        # examples["input_ids_two"] = tokens_two
        # return examples

        return example

# tokenizer = CLIPTokenizer.from_pretrained(
        # "stabilityai/stable-diffusion-2-base", subfolder="tokenizer", revision=None
    # )

# def collate_fn(examples):
#     input_ids = [example["instance_prompt_ids"] for example in examples]
#     pixel_values = [example["instance_images"] for example in examples]
#     reward = [example["reward"] for example in examples]

#     pixel_values = torch.stack(pixel_values)
#     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

#     input_ids = tokenizer.pad(
#         {"input_ids": input_ids},
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         return_tensors="pt",
#     ).input_ids

#     batch = {
#         "input_ids": input_ids,
#         "pixel_values": pixel_values,
#         "reward":reward
#     }
#     return batch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
    # input_ids_neg_one = torch.stack([example["input_ids_neg_one"] for example in examples])
    # input_ids_neg_two = torch.stack([example["input_ids_neg_two"] for example in examples])
    # match_score = [example["match_score"] for example in examples]
    # mismatch_score = [example["mismatch_score"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        # "input_ids_neg_one": input_ids_neg_one,
        # "input_ids_neg_two": input_ids_neg_two, 
        # "match_score": match_score, 
        # "mismatch_score": mismatch_score,
    }

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

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
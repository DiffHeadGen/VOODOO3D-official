from functools import cached_property
import torch

import cv2
import os.path as osp
from tqdm import tqdm
import yaml
import numpy as np

from data_preprocessing.data_preprocess import DataPreprocessor
from models import get_model
from utils.image_utils import tensor2img

from expdataloader import *


def tensor_from_path(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.0
    img = img * 2 - 1
    img = torch.from_numpy(img).float()
    return img


class VOODOOInfer:
    def __init__(self, skip_preprocess=False):
        self.device = "cuda"
        self.config_path = "configs/voodoo3d.yml"
        self.model_path = "pretrained_models/voodoo3d.pth"
        self.skip_preprocess = skip_preprocess

    @cached_property
    def processor(self):
        return DataPreprocessor(self.device)

    @cached_property
    def model(self):
        with open(self.config_path, "r") as f:
            options = yaml.safe_load(f)
        model = get_model(options["model"]).to(self.device)
        state_dict = torch.load(self.model_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        print(self.model_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def infer_data(self, source_data, driver_data):
        out = self.model(xs_data=source_data, xd_data=driver_data)
        out_hr = tensor2img(out["image"], min_max=(-1, 1))
        source_img = tensor2img(source_data["image"][0], min_max=(-1, 1))
        driver_img = tensor2img(driver_data["image"][0], min_max=(-1, 1))
        return source_img, driver_img, out_hr
        # cv2.imwrite(save_path, np.hstack((source_img, driver_img, out_hr)))

    def get_source_data(self, source_img_path):
        if not self.skip_preprocess:
            source_data = self.processor.from_path(source_img_path, self.device, keep_bg=False)
        else:
            source_data = {"image": tensor_from_path(source_img_path).to(self.device)}
        return source_data

    def get_driver_data(self, driver_img_path):
        if not self.skip_preprocess:
            driver_data = self.processor.from_path(driver_img_path, self.device, keep_bg=False)
            driver_data["exp_image"] = driver_data["image"]
        else:
            driver_data = {"exp_image": tensor_from_path(driver_img_path).to(self.device), "image": tensor_from_path(driver_img_path).to(self.device)}
        return driver_data

    def infer(self, source_img_path, driver_img_paths, save_path):
        source_data = self.get_source_data(source_img_path)
        for driver_img_path in driver_img_paths:
            driver_data = self.get_driver_data(driver_img_path)
            source_img, driver_img, out_hr = self.infer_data(source_data, driver_data)
            cv2.imwrite(save_path, np.hstack((source_img, driver_img, out_hr)))


class VOODOOLoader(RowDataLoader):
    def __init__(self, name="VOODOO3D"):
        super().__init__(name)

    @cached_property
    def model(self):
        return VOODOOInfer()

    def run_video(self, row):
        source_data = self.model.get_source_data(row.source_img_path)
        for driver_img_path in tqdm(row.target.img_paths):
            driver_data = self.model.get_driver_data(driver_img_path)
            source_img, driver_img, out_hr = self.model.infer_data(source_data, driver_data)
            data_name = osp.splitext(osp.basename(driver_img_path))[0]
            cv2.imwrite(f"{row.output.ori_output_comp_dir}/{data_name}.jpg", np.hstack((source_img, driver_img, out_hr)))
            cv2.imwrite(f"{row.output.ori_output_dir}/{data_name}.jpg", out_hr)
        row.output.merge_ori_output_comp_video()
        row.output.merge_ori_output_video()
        self.retarget_video(row)


def main():
    loader = VOODOOLoader()
    # row = loader.all_data_rows[0]
    # loader.run_video(row)
    loader.run_all()


if __name__ == "__main__":
    main()

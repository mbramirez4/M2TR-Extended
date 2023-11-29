import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import accuracy
from PIL import Image

import M2TR.utils.checkpoint as cu
import M2TR.utils.distributed as du
import M2TR.utils.logging as logging
from M2TR.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_loss_fun,
    build_model,
)
from M2TR.utils.env import pathmgr
from M2TR.utils.meters import AucMetric, MetricLogger

logger = logging.get_logger(__name__)

def create_custom_dataset_file(root, ds_name):
    dst = os.path.join(root, f"{ds_name}_splits_test.txt")
    paths = [path.replace(root, "").lstrip("/")
             for path in glob(os.path.join(root, "**", "*.*"), recursive=True)
             if "_splits_" not in path]
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write("\n".join(paths))

def infer(cfg, **kwargs):
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    root = cfg['DATASET']['ROOT_DIR']
    ds_name = cfg['DATASET']['DATASET_NAME']
    create_custom_dataset_file(root, ds_name)

    dataset = build_dataset('test', cfg)
    data_loader = build_dataloader(dataset, 'test', cfg)

    logger.info("Testing model for {} iterations".format(len(data_loader)))
    perform_inference(data_loader, model, cfg)

@torch.no_grad()
def perform_inference(
    data_loader, model, cfg, mode='Test'
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metric_logger = MetricLogger(delimiter="  ")
    header = mode + ':'

    model.eval()

    preds_list = []
    for samples in data_loader:
        img_paths = None
        if 'img_path' in samples:
            img_paths = samples.pop('img_path')
        
        samples = dict(
            zip(
                samples,
                map(
                    lambda sample: sample.to(device),
                    samples.values(),
                ),
            )
        )
        
        outputs = model(samples)
        preds = F.softmax(outputs['logits'], dim=1)[:, 1]

        if img_paths is None:
            continue
        for mask, img_path, pred in zip(outputs['mask'], img_paths, preds):
            dst_path = os.path.join("outputs", img_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            mask = mask.squeeze().cpu().numpy()
            mask = np.uint8(mask * 255)
            mask = Image.fromarray(mask)
            mask.save(dst_path)
            preds_list.append((img_path, pred.item()))
        print(f"Processed {len(preds_list)} images")
    
    with open("outputs/preds.txt", "w") as f:
        f.write("\n".join([f"{img_path} {pred}" for img_path, pred in preds_list]))


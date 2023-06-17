from time import time
from tqdm import trange
import torch
import numpy as np
import cv2
import os
import sys

def evaluate_dataset(dataset, predictor, metrics_hub, logger, save_dir=None):
    for sample_i in trange(len(dataset), desc=f'Testing on {metrics_hub.name}'):
        sample = dataset.get_sample(sample_i)
        sample = dataset.augment_sample(sample)

        sample_mask = sample['object_mask']
        pred = predictor.predict(sample['image'], sample_mask, return_numpy=False) # H,W,C


        target_image = torch.as_tensor(sample['target_image'], dtype=torch.float32).to(predictor.device)
        sample_mask = torch.as_tensor(sample_mask, dtype=torch.float32).to(predictor.device) #H,W
        string = ""
        with torch.no_grad():
            _,string = metrics_hub.compute_and_add(pred, target_image, sample_mask)
            logger.info(metrics_hub)

        # save images
        if save_dir!='':
            input_image = sample['image']
            pred = pred.detach().cpu().numpy().astype(np.uint8)
            target = sample['target_image']
            sample_mask = (sample_mask.unsqueeze(-1).repeat(1,1,3).detach().cpu().numpy()*255).astype(np.uint8)
            #out_image = np.concatenate([input_image,sample_mask, target, pred], axis=1)
            out_image = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            # metrics_list = metrics_hub.metrics
            # MSE = metrics_list[1]
            # fMSE = metrics_list[2]
            # PSNR = metrics_list[3]
            
            id,suffix = os.path.splitext(sample['image_id'])
            id = id.split("/")[-1]
            #print(os.path.join(save_dir, id+suffix))
            cv2.imwrite(os.path.join(save_dir, id+suffix), out_image ,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # cv2.imshow("out",out_image)
        # cv2.waitKey(0)


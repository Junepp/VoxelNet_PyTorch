import os
import argparse

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DataLoader
from loader.kitti import collate_fn

from model.model import RPN3D
from loader.kitti import KITTI as Dataset


parser = argparse.ArgumentParser(description='training')

parser.add_argument('--tag', type=str, default='default', help='log tag')
parser.add_argument('--input_path', type=str, default='/mnt/hdd_2T/ryan/dataset/sample')
parser.add_argument('--output_path', type=str, default='./preds', help='results output dir')
parser.add_argument('--vis', type=bool, default=True, help='set to True if dumping visualization')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--pretrained', type=str, default='/home/ryan/workspace/VoxelNet_PyTorch/saved/trial/060ep_loss0.2582.pt')


args = parser.parse_args()


def change_key(checkpoint):
    from collections import OrderedDict

    new = OrderedDict()

    for k, v in checkpoint.items():
        new_k = k.replace('module.', '')
        new[new_k] = v

    return new


def run():
    # Build data loader
    val_dataset = Dataset(os.path.join(args.input_path), shuffle=False, aug=False, is_testset=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=0, pin_memory=False)

    # Build model
    model = RPN3D('Car')

    # Resume model
    checkpoint = torch.load(args.pretrained)
    checkpoint = change_key(checkpoint)
    model.load_state_dict(checkpoint)

    model = nn.DataParallel(model).cuda()
    model.eval()

    with torch.no_grad():
        for (i, val_data) in enumerate(val_dataloader):
            # Forward pass for validation and prediction
            probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data)
            front_images, bird_views, heatmaps = None, None, None

            if args.vis:
                tags, ret_box3d_scores, front_images, bird_views, heatmaps = \
                    model.module.predict(val_data, probs, deltas, summary=False, vis=True)
            else:
                tags, ret_box3d_scores = model.module.predict(val_data, probs, deltas, summary=False, vis=False)

            # tags: (N)
            # ret_box3d_scores: (N, N'); (class, x, y, z, h, w, l, rz, score)

            # for tag, score in zip(tags, ret_box3d_scores):
            #     output_path = os.path.join(args.output_path, 'data', tag + '.txt')
            #
            #     with open(output_path, 'w+') as f:
            #         labels = box3d_to_label([score[:, 1:8]], [score[:, 0]], [score[:, -1]], coordinate='lidar')[0]
            #         for line in labels:
            #             f.write(line)
            #         print('Write out {} objects to {}'.format(len(labels), tag))

            # Dump visualizations
            if args.vis:
                for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                    front_img_path = os.path.join(args.output_path, 'vis', tag + '_front.jpg')
                    bird_view_path = os.path.join(args.output_path, 'vis', tag + '_bv.jpg')
                    heatmap_path = os.path.join(args.output_path, 'vis', tag + '_heatmap.jpg')

                    front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(front_img_path, front_image)
                    cv2.imwrite(bird_view_path, bird_view)
                    cv2.imwrite(heatmap_path, heatmap)


if __name__ == '__main__':
    save_model_dir = os.path.join('./saved', args.tag)

    # Create output folder
    os.makedirs(args.output_path, exist_ok=True)
    # os.makedirs(os.path.join(args.output_path, 'data'), exist_ok=True)

    if args.vis:
        os.makedirs(os.path.join(args.output_path, 'vis'), exist_ok=True)

    run()

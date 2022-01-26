# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
import numpy as np
from monai.inferers import sliding_window_inference
from networks.unetr import UNETR
from utils.data_utils import get_loader
from trainer import dice
import opts
import paddle.nn.functional as F


def main(args):
    args.test_mode = True
    args.data_type = "validation"
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        pos_embed=args.pos_embed,
        norm_name=args.norm_name,
        conv_block=True,
        res_block=True,
        dropout_rate=args.dropout_rate)
    model_dict = paddle.load(pretrained_pth)
    model.load_dict(model_dict['state_dict'])
    model.eval()

    with paddle.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"], batch["label"])
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            val_outputs = F.softmax(val_outputs, 1)
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            for i in range(1, 14):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))


if __name__ == '__main__':
    args = opts.eval_options()
    main(args)

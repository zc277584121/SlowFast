import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from slowfast.utils.parser import load_config, parse_args
from slowfast.visualization.utils import GetWeightAndActivation, process_layer_index_data
from collections import defaultdict
from slowfast.models import build_model
from torchvision.io import read_video


def normalization(data):
    range_ = np.max(data) - np.min(data)
    return (data - np.min(data)) / range_


def plot_input_sample(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()


def plot_attn_map(activations: dict, show=False):
    attn_map_dict = {}
    num_cols = len(activations.keys())
    max_head_num = 1
    for idx, layer_name in enumerate(activations.keys()):
        attn = activations[layer_name]
        head_num = attn.shape[1]
        if head_num > max_head_num:
            max_head_num = head_num
    num_rows = max_head_num
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_cols * 5, num_rows * 5))

    for idx, layer_name in enumerate(activations.keys()):
        attn = activations[layer_name]
        attn = attn.to('cpu').detach()
        # (B, head_num, Nq + 1, Nkv + 1)
        head_num = attn.shape[1]

        for head_idx in range(max_head_num):
            if head_idx < head_num:
                # 取了batch里第一个样本
                one_head_attn = attn[0][head_idx]

                Nkv = one_head_attn.shape[-1]
                # Nkv + 1
                one_head_attn_without_cls_pos = one_head_attn[1:]
                # Nq, Nkv + 1
                thw = one_head_attn_without_cls_pos.shape[0] // 4
                h = w = int(math.sqrt(thw))
                out_shape = (4, h, w)
                one_head_attn_without_cls_pos = one_head_attn_without_cls_pos.reshape(*out_shape, Nkv)
                # 4, H, W, Nkv + 1

                one_head_attn_without_cls_pos_t = one_head_attn_without_cls_pos[1]
                # H, W, Nkv + 1
                # 取了第2个时间片
                attn_tensor = one_head_attn_without_cls_pos_t[..., 0]
                # 取了Nkv 维度上的第0个位置，即cls_token
                # H, W

                attn_np = normalization(attn_tensor.numpy())
                cm = plt.get_cmap("viridis")
                heatmap = cm(attn_np)
                heatmap = heatmap[:, :, :3]
                # Convert (H, W, C) to (C, H, W)

                axs[head_idx, idx].imshow(heatmap)
                axs[head_idx, idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                name = layer_name + '/head_' + str(head_idx)
                axs[head_idx, idx].set_title(name)
                attn_map_dict[name] = heatmap
            else:
                axs[head_idx, idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    if show:
        plt.show()
    return attn_map_dict


def plot_feature_map(activations: dict, thws: dict, show=False):
    feature_map_dict = {}
    num_cols = len(activations.keys())
    cm = plt.get_cmap("viridis")
    _, axs = plt.subplots(nrows=1, ncols=num_cols, squeeze=False, figsize=(30, 5))
    for idx, layer_name in enumerate(activations.keys()):
        output = change_shape(input_tensor=activations[layer_name], thw=thws[layer_name])
        output = output.to('cpu').numpy()
        output = normalization(output)
        heatmap = cm(output)
        heatmap = heatmap[:, :, :3]
        axs[0, idx].imshow(heatmap)
        axs[0, idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, idx].set_title(layer_name)
        feature_map_dict[layer_name] = heatmap
    plt.tight_layout()
    if show:
        plt.show()
    return feature_map_dict


def change_shape(input_tensor: torch.Tensor, thw: list):
    input_tensor = input_tensor[0]
    C = input_tensor.shape[-1]
    input_tensor = input_tensor[1:]  ## (3136, 384) (THW, C)
    input_tensor = input_tensor.view(*thw, C)  # (4, 28, 28, 384) (T, H, W, C)
    input_tensor = input_tensor[0]  # (28, 28, 384)
    input_tensor = input_tensor[..., 0]  # (28, 28)
    return input_tensor


def split_acts_and_thws(activations: dict, thws: dict):
    feature_map_activations = {}
    feature_map_thws = thws.copy()
    attn_map_activations = {}
    attn_thws = {}
    for layer_name in activations.keys():
        if layer_name in thws.keys():
            feature_map_activations[layer_name] = activations[layer_name]
        else:
            attn_map_activations[layer_name] = activations[layer_name]
    return feature_map_activations, feature_map_thws, attn_map_activations, attn_thws


def get_plot_map_dict(video_path, cfg, sec=0, sample_f_delta=1, show=False):
    sample_num = 8

    end_sec = sec + 1
    origin_frames, _, meta = read_video(str(video_path), start_pts=sec, end_pts=end_sec, pts_unit='sec')
    fps = meta['video_fps']
    origin_frames = origin_frames.permute(0, 3, 1, 2)
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.45, std=0.225),
            T.CenterCrop(size=(224, 224))
        ]
    )
    show_transforms = T.Compose(
        [
            T.CenterCrop(size=(224, 224))
        ]
    )

    frames = transforms(origin_frames)
    show_frames = show_transforms(origin_frames)
    index_list = list(range(0, sample_f_delta * sample_num, sample_f_delta))
    img_batch = torch.stack([frames[index_list]])
    show_stack = torch.stack([show_frames[index_list]])
    plot_input_sample(show_stack.squeeze(0))
    model = build_model(cfg)
    model.eval()
    img_batch = img_batch.transpose(1, 2)
    img_batch = img_batch.to('cuda')
    img_batch = [img_batch]

    n_devices = cfg.NUM_GPUS * cfg.NUM_SHARDS
    prefix = "module/" if n_devices > 1 else ""

    layer_ls, indexing_dict = process_layer_index_data(
        cfg.TENSORBOARD.MODEL_VIS.LAYER_LIST, layer_name_prefix=prefix
    )

    model_vis = GetWeightAndActivation(model, layer_ls)
    activations, _, thws = model_vis.get_activations(img_batch)

    feature_map_activations, feature_map_thws, attn_map_activations, attn_thws = split_acts_and_thws(activations, thws)

    feature_map_dict = {}
    attn_map_dict = {}
    if len(feature_map_activations) > 0:
        feature_map_dict = plot_feature_map(feature_map_activations, feature_map_thws, show=show)

    if len(attn_map_activations) > 0:
        attn_map_dict = plot_attn_map(attn_map_activations, show=show)

    return feature_map_dict, attn_map_dict


def get_frame_by_sec(video_path, sec):
    origin_frames, _, meta = read_video(str(video_path), start_pts=sec, end_pts=sec + 0.1, pts_unit='sec')
    origin_frames = origin_frames.permute(0, 3, 1, 2)
    origin_frames = torch.unsqueeze(origin_frames[0], dim=0)

    show_transforms = T.Compose(
        [
            T.CenterCrop(size=(224, 224))
        ]
    )
    origin_frames = show_transforms(origin_frames)

    img = F.to_pil_image(origin_frames[0].to("cpu"))
    np_img = np.asarray(img)
    return np_img


args = parse_args()
args.cfg_file = 'configs/Kinetics/MVIT_B_16x4_CONV.yaml'
args.num_shards = 1
args.init_method = 'tcp://localhost:9999'
args.shard_id = 0
args.opts = ['DATA.PATH_TO_DATA_DIR', './', 'TEST.CHECKPOINT_FILE_PATH', 'K400_MVIT_B_16x4_CONV.pyth',
             'TRAIN.ENABLE', 'False', 'TEST.ENABLE', 'False']

cfg = load_config(args)

video_path = args.video_path  # = './q1IiVYEOS9U.mp4'
start_sec = args.start_sec  # 0
end_sec = args.end_sec  # 1.5
sec_delta = args.sec_delta  # 0.5
show_img = args.show_img
save_gif = args.save_gif
if end_sec is None:
    sec_list = [start_sec]
else:
    sec_list = np.arange(start_sec, end_sec, sec_delta)
# print('sec_list = ', sec_list)

original_img_list = []
show_list_dict = defaultdict(list)
fig = plt.figure()
ax = plt.gca()


def update_original(frame):
    ax.clear()
    art = original_img_list[frame % len(original_img_list)]
    ax.imshow(art)


def get_update_fun(layer_name):
    def update_show_img(frame):
        ax.clear()
        show_list = show_list_dict[layer_name]
        art = show_list[frame % len(show_list)]
        ax.imshow(art)

    return update_show_img


for sec in sec_list:
    original_img = get_frame_by_sec(video_path, sec)

    original_img_list.append(original_img)
    feature_map_dict, attn_map_dict = get_plot_map_dict(video_path, cfg=cfg, sec=sec, show=show_img)
    show_map_dict = {**feature_map_dict, **attn_map_dict}
    for layer_name in show_map_dict:
        show_list_dict[layer_name].append(show_map_dict[layer_name])

if save_gif:
    for layer_name in show_list_dict:
        print('anima.. ' + layer_name)
        anim = animation.FuncAnimation(fig, get_update_fun(layer_name), frames=int((end_sec - start_sec) / sec_delta),
                                       interval=sec_delta * 1000)

        file_name = layer_name.replace('/', '_') + '.gif'
        anim.save(filename=file_name)

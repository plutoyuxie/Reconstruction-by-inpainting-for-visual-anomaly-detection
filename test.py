import os
import argparse
import matplotlib
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter
from models.unet import UNet
from utils.gen_mask import gen_mask
from losses.gms_loss import MSGMS_Score
from datasets.mvtec import MVTecDataset
from utils.funcs import denormalization

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
plt.switch_backend('agg')


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--obj', type=str, default='bottle')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='./mvtec/bottle/seed_3338/bottle_2020-11-11-4859_model.pt')
    parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3338)
    parser.add_argument('--ratio', type=float, default=95)
    parser.add_argument('--k_value', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model and dataset
    args.input_channel = 1 if args.grayscale else 3
    model = UNet().to(device)
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['model'])

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    scores, test_imgs, recon_imgs, gt_list, gt_mask_list = test(args, model, test_loader)
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    plt.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (args.obj, img_roc_auc))
    plt.legend(loc="lower right")

    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

    plt.plot(fpr, tpr, label='%s pixel_ROCAUC: %.3f' % (args.obj, per_pixel_rocauc))
    plt.legend(loc="lower right")
    save_dir = args.save_dir + '/' + f'seed_{args.seed}' + '/' + 'pictures_{:.4f}'.format(threshold)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, args.obj + '_roc_curve.png'), dpi=100)

    plot_fig(args, test_imgs, recon_imgs, scores, gt_mask_list, threshold, save_dir)


def test(args, model, test_loader):
    model.eval()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    msgms_score = MSGMS_Score()
    for (data, label, mask) in tqdm(test_loader):
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())
        score = 0
        with torch.no_grad():
            data = data.to(device)
            for k in args.k_value:
                img_size = data.size(-1)
                N = img_size // k
                Ms_generator = gen_mask([k], 3, img_size)
                Ms = next(Ms_generator)
                inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
                outputs = [model(x) for x in inputs]
                output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
                score += msgms_score(data, output) / (N**2)

        score = score.squeeze().cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=7)
        scores.extend(score)
        recon_imgs.extend(output.cpu().numpy())
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list


def plot_fig(args, test_img, recon_imgs, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, args.obj + '_{}_png'.format(i)), dpi=100)
        plt.close()


if __name__ == '__main__':
    main()

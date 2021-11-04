import os, sys
import re

import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

parser = ArgumentParser()
parser.add_argument("--config", help="path to config")
parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
parser.add_argument("--result_video", default='result.mp4', help="path to output")

parser.add_argument("--relative", dest="relative", action="store_true",
                    help="use relative or absolute keypoint coordinates")
parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                    help="adapt movement scale based on convex hull of keypoints")

parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                    help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                    help="Set frame to start from.")

parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

parser.set_defaults(relative=True)
parser.set_defaults(adapt_scale=True)
parser.set_defaults(checkpoint='checkpoint/vox-adv-cpk.pth.tar')
parser.set_defaults(config='config/vox-adv-256.yaml')

args = parser.parse_args()


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def set_args(message):
    args.source_image = os.path.join(user_image, message['user_img'])
    args.driving_video = os.path.join(user_video, message['user_video'])
    args.result_video = os.path.join(result_path, re.sub("\.jpg|\.png|\.jpeg|\.mp4","",message['user_img']+"_"+message['user_video'])+".mp4")
    args.relative = message['relative']
    args.adapt_scale = message['adapt_scale']
    args.checkpoint = os.path.join(model_dir, "vox-adv-cpk.pth.tar")
    args.config = os.path.join(config_dir, "vox-adv-256.yaml")


model_dir = './checkpoint'  # 模型位置（目前只有一个，就不修改了）
config_dir = './config'  # 模型的配置文件位置（目前只有一个，就不修改了）

# user_image = './images'  # 黑白图片文件夹
# user_video = './video'  # 指导video
# result_path = './result'  # 输出文件夹
# message_json = './message.json'

user_image = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/firstOrder/user_imgs'  # 黑白图片文件夹
user_video = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/firstOrder/user_videos'  # 指导video
result_path = '/workspace/go_proj/src/Ai_WebServer/static/algorithm/firstOrder/res_videos'  # 输出文件夹
message_json = '/workspace/go_proj/src/Ai_WebServer/algorithm_utils/firstOrder/message.json'


if __name__ == '__main__':

    generator, kp_detector = None, None
    last_message = {}
    last_model = ""
    import time, json

    while True:
        try:
            with open(message_json, "r", encoding="utf-8") as f:
                message = json.load(f)
        except Exception as e:
            print(e)
            continue
        if message == last_message:
            print('waiting')
            time.sleep(1)
            continue
        else:
            set_args(message)
            last_message = message
            if last_model != args.checkpoint:
                generator, kp_detector = load_checkpoints(config_path=args.config, checkpoint_path=args.checkpoint, cpu=args.cpu)
                print('model reloaded!')
                last_model = args.checkpoint
        if os.path.exists(args.result_video):
            continue

        print('start processing!')
        source_image = imageio.imread(args.source_image)
        reader = imageio.get_reader(args.driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]


        if args.find_best_frame or args.best_frame is not None:
            i = args.best_frame if args.best_frame is not None else find_best_frame(source_image, driving_video, cpu=args.cpu)
            print("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector,
                                                 relative=args.relative, adapt_movement_scale=args.adapt_scale, cpu=args.cpu)
            predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector,
                                                  relative=args.relative, adapt_movement_scale=args.adapt_scale, cpu=args.cpu)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=args.relative,
                                         adapt_movement_scale=args.adapt_scale, cpu=args.cpu)
        imageio.mimsave(args.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        print('video saved!')

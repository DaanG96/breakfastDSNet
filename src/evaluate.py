import logging
from pathlib import Path

import numpy as np
import torch
import re
import json

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

logger = logging.getLogger()


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')
    json_file = []
    with torch.no_grad():
        for test_key, seq, gt, cps, n_frames, nfps, picks, user_summary, name in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
            pred_cls, pred_bboxes = model.predict(seq_torch)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ, score = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)
            pred_arr, pred_seg = convert_array(pred_summ, nfps)
            pred_summ = vsumm_helper.downsample_summ(pred_summ)
            json_file.append({"video":str(name), "gt": convert_array_2(gt), 
            "pred_score": convert_array_2(score), 
            "user_anno":convert_user(user_summary),
            "fscore": float(fscore),
            "pred_sum": convert_array_2(pred_summ)})
            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)

    return stats.fscore, stats.diversity, json_file

def convert_user(arr):
    res = []
    for i in arr:
        temp = []
        for a in i:
            temp.append(a.item())
        res.append(temp)
    return res

def convert_array_2(arr):
    res = []
    for i in arr:
        res.append(i.item())
    return res

def convert_array(user, nfps):
    user_arr = []
    shots_arr = []
    for b in user:
        user_arr.append(1 if b else 0)
    shots_arr.append(nfps[0].item())
    for i in range(1, len(nfps)):
        shots_arr.append(shots_arr[i-1] + nfps[i].item())
    return user_arr, shots_arr

def get_file_name(name):
    arr = re.split("[\\/]", name)
    print(arr)
    return arr[-1]


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    f = []
    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)
            fscore, diversity, json_file = evaluate(model, val_loader, args.nms_thresh, args.device)
            f += json_file
            stats.update(fscore=fscore, diversity=diversity)

            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')
    # with open('aftvsum.json', 'w') as fout:
    #     json.dump(f, fout)

if __name__ == '__main__':
    main()

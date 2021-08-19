import os.path as osp
import os
import numpy as np
import fire
import shutil

# Modify here
AP_ROOT = 'C:/Users/Philip Fu/datasets/ap'
SEQ_ROOT = osp.join(AP_ROOT, 'images/train')

def main(ap_root=AP_ROOT, seq_root=SEQ_ROOT):
    label_root = osp.join(ap_root, 'obs', 'det', 'train')
    os.makedirs(label_root, exist_ok=True)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'det', 'det.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        if os.path.exists(seq_label_root):
            shutil.rmtree(seq_label_root, ignore_errors=True)
        os.makedirs(seq_label_root, exist_ok=True)

        for fid, tid, x, y, w, h, mark, _, _, _ in gt:
            if mark == 0:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '{:.6f} {:.6f} {:.6f} {:.6f} 1.0\n'.format(
                x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

if __name__ == '__main__':
    fire.Fire(main)
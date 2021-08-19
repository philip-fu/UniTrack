import os
import pdb
import cv2
import time
import itertools
import copy
import os.path as osp
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import ops

from model import AppearanceModel,partial_load
from utils.log import logger
from core.association import matching
from core.propagation import propagate
from core.motion.kalman_filter import KalmanFilter

from utils.box import * 
from utils.mask import *
from .basetrack import *


class AssociationTracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(opt.frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        self.app_model = AppearanceModel(opt).to(opt.device)
        self.app_model.eval()
        
        if not self.opt.asso_with_motion:
            self.opt.motion_lambda = 1
            self.opt.motion_gated = False
        
    def extract_emb(self, img, obs):
        raise NotImplementedError

    def prepare_obs(self, img, img0, obs):
        raise NotImplementedError

    def update(self, img, img0, obs):
        torch.cuda.empty_cache()
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
 
        t1 = time.time()
        detections = self.prepare_obs(img, img0, obs)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        tracks = joint_stracks(tracked_stracks, self.lost_stracks)

        #for trk in tracks:
        #    if trk.track_id in (11, 14):
        #        print("track id {}, state: {}, activated: {}. Frame id {}, end frame {}".format(trk.track_id, trk.state, trk.is_activated, trk.frame_id, trk.end_frame))


        dists, recons_ftrk = matching.reconsdot_distance(tracks, detections)
        if self.opt.use_kalman: 
            # Predict the current location with KF
            STrack.multi_predict(tracks)
            dists = matching.fuse_motion(self.kalman_filter, dists, tracks, detections, 
                    lambda_=self.opt.motion_lambda, gate=self.opt.motion_gated)
        if len(obs) > 0 and self.opt.category_gated and obs.shape[1] == 6:
            dists = matching.category_gate(dists, tracks, detections)
        if len(obs) > 0 and self.opt.area_gated:
            dists = matching.area_gate(dists, tracks, detections, self.opt.area_diff_threshold)
        if len(obs) > 0 and self.opt.reactivate_on_edge:
            dists = matching.edge_gate(dists, tracks, detections, im_shape=self.opt.img_size[::-1])

        if int(os.getenv('DEBUG', 0)):
            PRINT_FRAME = os.getenv('PRINT_FRAME', None)
            if PRINT_FRAME is None:
                print('='*30)
                print(self.frame_id, dists)
            elif int(PRINT_FRAME) in [self.frame_id, -2]:
                print('='*30)
                print(self.frame_id, dists)
                

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.assign_thres) # was 0.7

        for itracked, idet in matches:
            track = tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                reactivate = True
                if self.opt.reactivate_on_edge:
                    if is_box_on_edge(track.tlbr, self.opt.img_size[::-1]):
                        if not is_box_on_edge(det.tlbr, self.opt.img_size[::-1]):
                            reactivate = False
                if reactivate:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
        
        if self.opt.use_kalman:
            '''(optional) Step 3: Second association, with IOU'''
            tracks = [tracks[i] for i in u_track if tracks[i].state==TrackState.Tracked]
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(tracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7) # was 0.5
            
            for itracked, idet in matches:
                track = tracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                    dists, thresh=self.opt.confirm_iou_thres)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        for it in u_track:
            track = tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            #track.activate(self.kalman_filter, self.frame_id)
            track.activate(KalmanFilter(), self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.removed_stracks = sub_stracks(self.removed_stracks, refind_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks, ioudist=self.opt.dup_iou_thres)

        #print_strack_status(self.tracked_stracks, 'self.tracked_stracks at the end of frame {}'.format(self.frame_id))
        #print_strack_status(self.lost_stracks, 'self.lost_stracks at the end of frame {}'.format(self.frame_id))
        

        # get scores of lost tracks
        output_stracks = [copy.copy(track) for track in self.tracked_stracks if track.is_activated]
        return output_stracks



        

def print_strack_status(stracks, name):
    print(name)
    ids = []
    for trk in stracks:
        ids.append(trk.track_id)

    print(ids)
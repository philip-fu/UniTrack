import pdb
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import lap

from cython_bbox import bbox_overlaps as bbox_ious
from core.motion import kalman_filter
from utils.box import is_box_on_edge

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
            

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def dimension_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlwhs = atracks
        btlwhs = btracks
    else:
        atlwhs = [track.tlwh for track in atracks]
        btlwhs = [track.tlwh for track in btracks]
    cost_matrix = dimension_diff(atlwhs, btlwhs)
    
    return cost_matrix

def dimension_diff(atlwhs, btlwhs):
    dist = np.zeros((len(atlwhs), len(btlwhs)), dtype=np.float)
    if dist.size == 0:
        return dist

    dist = cdist(
        np.array(atlwhs)[:,2:4],
        np.array(btlwhs)[:,2:4],
    )

    return dist

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features)) # Nomalized features
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98, gate=True):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 4 if only_position else 8
    gating_threshold = kalman_filter.chi2inv95[gating_dim] * 10

    dimension_cost_matrix = dimension_distance(tracks, detections)

    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        if gate:
            cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = cost_matrix[row] + (1-lambda_) * 1. * np.sqrt(gating_distance) + (1-lambda_) * .0 * dimension_cost_matrix[row]
    return cost_matrix


def center_emb_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = torch.stack([track.curr_feat.squeeze() for track in detections])
    track_features = torch.stack([track.smooth_feat.squeeze() for track in tracks])
    normed_det = F.normalize(det_features)
    normed_track = F.normalize(track_features)
    cost_matrix = torch.mm(normed_track, normed_det.T)
    cost_matrix = 1 - cost_matrix.detach().cpu().numpy()
    return cost_matrix

def recons_distance(tracks, detections, tmp=100):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features_ = torch.stack([track.curr_feat.squeeze() for track in detections])
    track_features_ = torch.stack([track.smooth_feat for track in tracks])
    det_features = F.normalize(det_features_, dim=1)
    track_features = F.normalize(track_features_, dim=1)

    ndet, ndim, nw, nh = det_features.shape
    ntrk, _, _, _ = track_features.shape
    fdet = det_features.permute(0,2,3,1).reshape(-1, ndim).cuda()        # ndet*nw*nh, ndim
    ftrk = track_features.permute(0,2,3,1).reshape(-1, ndim).cuda()      # ntrk*nw*nh, ndim

    aff = torch.mm(ftrk, fdet.transpose(0,1))                             # ntrk*nw*nh, ndet*nw*nh
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0,1)

    recons_ftrk = torch.einsum('tds,dsm->tdm', aff_td.view(ntrk*nw*nh, ndet, nw*nh), 
                                fdet.view(ndet, nw*nh, ndim))         # ntrk*nw*nh, ndet, ndim
    recons_fdet = torch.einsum('dts,tsm->dtm', aff_dt.view(ndet*nw*nh, ntrk, nw*nh),
                                ftrk.view(ntrk, nw*nh, ndim))         # ndet*nw*nh, ntrk, ndim
 
    res_ftrk = (recons_ftrk.permute(0,2,1) - ftrk.unsqueeze(-1)).view(ntrk, nw*nh*ndim, ndet)
    res_fdet = (recons_fdet.permute(0,2,1) - fdet.unsqueeze(-1)).view(ndet, nw*nh*ndim, ntrk)

    cost_matrix = (torch.abs(res_ftrk).mean(1) + torch.abs(res_fdet).mean(1).transpose(0,1)) * 0.5
    cost_matrix = cost_matrix / cost_matrix.max(1)[0].unsqueeze(-1) 
    #pdb.set_trace()
    cost_matrix = cost_matrix.cpu().numpy()
    return cost_matrix


def get_track_feat(tracks, feat_flag='curr'):
    if feat_flag == 'curr':
        feat_list = [track.curr_feat.squeeze(0) for track in tracks]
    elif feat_flag == 'smooth':
        feat_list = [track.smooth_feat.squeeze(0) for track in tracks]
    else:
        raise NotImplementedError
    
    n = len(tracks)
    fdim = feat_list[0].shape[0]
    fdim_num = len(feat_list[0].shape)
    if fdim_num > 2:
        feat_list = [f.view(fdim,-1) for f in feat_list]
    numels = [f.shape[1] for f in feat_list]
    
    ret = torch.zeros(n, fdim, np.max(numels)).to(feat_list[0].device)
    for i, f in enumerate(feat_list):
        ret[i, :, :numels[i]] = f
    return ret 

def reconsdot_distance(tracks, detections, tmp=100):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix, None
    det_features_ = get_track_feat(detections)
    track_features_ = get_track_feat(tracks, feat_flag='curr')

    det_features = F.normalize(det_features_, dim=1)
    track_features = F.normalize(track_features_, dim=1)

    ndet, ndim, nsd = det_features.shape
    ntrk, _, nst = track_features.shape

    fdet = det_features.permute(0, 2, 1).reshape(-1, ndim).cuda()
    ftrk = track_features.permute(0, 2, 1).reshape(-1, ndim).cuda()

    aff = torch.mm(ftrk, fdet.transpose(0, 1))
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0, 1)

    recons_ftrk = torch.einsum('tds,dsm->tdm', aff_td.view(ntrk*nst, ndet, nsd),
                               fdet.view(ndet, nsd, ndim))
    recons_fdet = torch.einsum('dts,tsm->dtm', aff_dt.view(ndet*nsd, ntrk, nst),
                               ftrk.view(ntrk, nst, ndim))

    recons_ftrk = recons_ftrk.permute(0, 2, 1).view(ntrk, nst*ndim, ndet)
    recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    recons_fdet = recons_fdet.permute(0, 2, 1).view(ndet, nsd*ndim, ntrk)
    recons_fdet_norm = F.normalize(recons_fdet, dim=1)

    dot_td = torch.einsum('tad,ta->td', recons_ftrk_norm,
                          F.normalize(ftrk.reshape(ntrk, nst*ndim), dim=1))
    dot_dt = torch.einsum('dat,da->dt', recons_fdet_norm,
                          F.normalize(fdet.reshape(ndet, nsd*ndim), dim=1))

    cost_matrix = 1 - 0.5 * (dot_td + dot_dt.transpose(0, 1))
    cost_matrix = cost_matrix.detach().cpu().numpy()

    return cost_matrix, None


def category_gate(cost_matrix, tracks, detections):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix

    det_categories = np.array([d.category for d in detections])
    trk_categories = np.array([t.category for t in tracks])

    cost_matrix = cost_matrix + np.abs(
            det_categories[None, :] - trk_categories[:, None])
    return cost_matrix

def area_gate(cost_matrix, tracks, detections, gated_threshold):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param gated_threshold: float between [0,1]. Lower bound of area diff
    :return: cost_matrix np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix

    area_diff_matrix = area_distance(tracks, detections)


    cost_matrix[np.where(area_diff_matrix < gated_threshold)] = np.inf
    return cost_matrix


def area_distance(atracks, btracks):
    """
    Compute cost based on areaA / areaB, where areaB > areaA
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlwhs = atracks
        btlwhs = btracks
    else:
        atlwhs = np.stack([track.tlwh for track in atracks], axis=0)
        btlwhs = np.stack([track.tlwh for track in btracks], axis=0)
    cost_matrix = area_diff(atlwhs, btlwhs)
    
    return cost_matrix

def area_diff(atlwhs, btlwhs):
    dist = np.zeros((len(atlwhs), len(btlwhs)), dtype=np.float)
    if dist.size == 0:
        return dist

    dist = cdist(
        np.array(atlwhs[:, 2] * atlwhs[:, 3]).reshape((-1,1)),
        np.array(btlwhs[:, 2] * btlwhs[:, 3]).reshape((-1,1)),
        lambda u, v: min(u/max(1e-5,v), v/max(1e-5,u))
    )

    return dist


def edge_gate(cost_matrix, tracks, detections, im_shape, penalty=1.25):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param penalty: float. If i at edge but j not at edge, cost of i and j will multiply penalty
    :return: cost_matrix np.ndarray. If both the edge status (at image edge or not) of i,j is the same or if they have iou>0, cost_matrix[i,j]=1
    """
    if cost_matrix.size == 0:
        return cost_matrix

    edge_matrix = edge_distance(tracks, detections, im_shape)
    cost_matrix[np.where(edge_matrix == 0)] *= penalty
    return cost_matrix


def edge_distance(atracks, btracks, im_shape):
    """
    Compute cost based on edge status
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = np.stack([track.tlbr for track in atracks], axis=0)
        btlbrs = np.stack([track.tlbr for track in btracks], axis=0)

    cost_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    cost_matrix = cdist(
        atlbrs,
        btlbrs,
        lambda u, v: (is_box_on_edge(u, im_shape) == is_box_on_edge(v, im_shape))
    )

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    cost_matrix += ious
    
    return cost_matrix


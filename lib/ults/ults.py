import torch
import torch.nn as nn



def Matching_OBJ(entry, gt_annotation, mode='sgcls'):

    match_node = []
    
    obj_dis = entry["distribution"]
    boxes = entry['boxes']
    obj_feat_similarity = nn.functional.cosine_similarity(obj_dis.unsqueeze(1), obj_dis.unsqueeze(0), dim=-1)
    iou = compute_iou(boxes, boxes)
    score = obj_feat_similarity + iou 
  
    for obj_id in range(entry['features'].shape[0]):
        f_id = entry['boxes'][:,0][obj_id]
        indices = torch.tensor([]).cuda()
        for i in range(len(gt_annotation)):
            if f_id == i:
                continue  # 物体匹配时排除同一帧的物体
            else:
                m_id = torch.nonzero(entry['boxes'][:,0] == i).squeeze(1)
                p_value, p_index = torch.max(score[obj_id, m_id], dim=0)
                indices = torch.cat([indices, m_id[p_index].unsqueeze(0)], dim=0)
        match_node.append(indices)
    
    entry['match_node'] = match_node
    
    return entry


def compute_iou(ibox_f, ibox_n):
    """
        Calculate IoU between two sets of bounding boxes.

        Args:
            boxes1: a tensor of shape (N, 4) representing N boxes
            boxes2: a tensor of shape (M, 4) representing M boxes

        Returns:
            iou: a tensor of shape (N, M) representing the IoU between each pair of boxes
        """
    # 定义 box1 和 box2
    boxes1 = ibox_f[:, 1:]
    boxes2 = ibox_n[:, 1:]

    # 计算相交部分的左上角和右下角坐标
    intersection_topleft = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # shape: (N, M, 2)
    intersection_bottomright = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # shape: (N, M, 2)
    # 计算相交部分的宽度和高度
    intersection_wh = torch.clamp(intersection_bottomright - intersection_topleft, min=0)  # shape: (N, M, 2)
    # 计算相交部分的面积
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]  # shape: (N, M)
    # 计算两组框分别占据的总面积
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # shape: (N,)
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # shape: (M,)
    # 计算 IoU
    union_area = boxes1_area[:, None] + boxes2_area - intersection_area
    obj_boxes_iou = intersection_area / union_area

    return obj_boxes_iou



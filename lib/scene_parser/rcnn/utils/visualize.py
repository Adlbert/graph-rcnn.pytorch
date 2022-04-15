import cv2
import torch
import numpy as np
import networkx as nx
from itertools import compress
import hashlib

def select_top_pred_predictions(predictions, confidence_threshold=0.0):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score
    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.
    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    scores = scores.max(1)[0]
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = scores[keep]
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

#origianl threshold = 0.2
def select_top_predictions(predictions, confidence_threshold=0.0):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score
    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.
    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def compute_colors_for_labels(labels, palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def compute_colors_for_idx_pairs(idx_pairs, palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])):
    """
    Simple function that adds fixed colors depending on the class
    """
    temp = torch.ones((idx_pairs.size(0), 1),  dtype=torch.int64)
    idx_pairs = torch.cat((temp, idx_pairs), 1)
    colors = idx_pairs[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

    
def overlay_lines(image, predictions):
    idx_pairs = predictions.get_field("idx_pairs").tolist()
    boxes = predictions.bbox

    for box in boxes:
        w1, h1 = np.abs(box[:2] - box[2:4])
        x1, y1 = (box[0] + w1/2, box[1] + h1/2)
        x1 = round(x1.item())
        y1 = round(y1.item())

        w2, h2 = np.abs(box[4:6] - box[6:8])
        x2, y2 = (box[4] + w2/2, box[5] + h2/2)
        x2 = round(x2.item())
        y2 = round(y2.item())

        start = (x1, y1)
        end = (x2,y2)
        cv2.line(image, start, end, color=(255,255,255), thickness=1)
    return image

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_class_names(image, predictions, categories):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    colors = compute_colors_for_labels(predictions.get_field("labels")).tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions.bbox



    template = "{}: {:.2f}"
    for box, score, label, color in zip(boxes, scores, labels, colors):
        x, y = box[:2]
        x = round(x.item())
        y = round(y.item()+20)
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(color), 1, cv2.LINE_AA
        )

    return image

def overlay_question_answers(image, qas, max_num=10):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    height, width = image.shape[0], image.shape[1]
    template = "Question: {} \t Answer: {}"
    for i, qa in enumerate(qas):
        x, y = 20, (height - 20 - 20 * i)
        s = template.format(qa[0], qa[1])
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1
        )
    return image

def hash_tensor(tesnor):
    hash = ""
    for i in tesnor:
        hash += str(i.item())
    return hash

def generate_graph(img_id, predictions, pred_predictions, categories, predicates):
    G = nx.DiGraph()
    labeldict = {}
    edge_labeldict = {}

    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions.bbox
    LABELS_FILTER = ['man', 'face', 'woman', 'hand', 'head'] 
    # LABELS_FILTER = labels

    
    for num, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if label in LABELS_FILTER:
            box = hash_tensor(box.int())
            G.add_node(num)
            labeldict[num] = label

    pred_scores = pred_predictions.get_field("scores")
    idx_pairs = pred_predictions.get_field("idx_pairs").tolist()

    for idx_pair, pred_score in zip(idx_pairs, pred_scores):
        idx_0 = idx_pair[0]
        idx_1 = idx_pair[1]
        add = True
        if idx_0 in labeldict and labeldict[idx_0] in LABELS_FILTER and idx_1 in labeldict and labeldict[idx_1] in LABELS_FILTER:
            m = torch.max(pred_score[0:51])
            if m.item() > 0.0:
                index = (pred_score == m).nonzero().flatten()
                p = predicates[index]
                key = (idx_0,idx_1)

                this_start_label = labels[idx_0]
                this_end_label = labels[idx_1]
                for edge in list(G.edges):
                    start = edge[0]
                    end = edge[1]
                    if idx_1 == end:
                        start_label = labels[start]
                        end_label = labels[end]
                        edge_label = edge_labeldict[(start,end)]
                        predicate = edge_label.split(': ')[0]
                        score = float(edge_label.split(': ')[1])
                        if start_label == this_start_label and end_label == this_end_label and predicate == p:
                            add = score < m.item()
                            if add:
                                G.remove_edge(start, end)

                if add:
                    G.add_edge(idx_0, idx_1)
                    edge_labeldict[key] = "{}: {}".format(p,round(m.item(),2))

    return G, labeldict, edge_labeldict




def generate_top_graph(img_id, predictions, pred_predictions, categories, predicates):
    G_in = nx.DiGraph()
    G_out = nx.DiGraph()
    labeldict = {}
    edge_labeldict = {}

    LABELS_FILTER = ['man', 'face', 'woman', 'hand', 'head'] 
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [categories[i] for i in labels]
    label_mask = [l in LABELS_FILTER for l in labels]
    # labels = list(compress(labels,label_mask))
    boxes = predictions.bbox
    # boxes = list(compress(boxes,label_mask))

    for num, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        box = hash_tensor(box.int())
        G_in.add_node(box)
        G_out.add_node(box)
        labeldict[box] = label


    pred_scores = pred_predictions.get_field("scores")
    # pred_scores = pred_scores.max(1)[0].tolist()
    if len(pred_scores) == 0:
        return G_in, labeldict, edge_labeldict
    idx_pairs = pred_predictions.get_field("idx_pairs").tolist()
    named_idx_pairs = []
    for i in idx_pairs:
        if i[0] + 1 > len(predicates) or i[1]+1 > len(predicates):
            #Sometimes too large idenxes are computed.
            continue
        named_idx_pairs.append([predicates[i[0]], predicates[i[1]]])
    pred_boxes = pred_predictions.bbox

    pred_score_dict = dict()
    n  = 0
    calls  = 0
    l_1 = list()
    l_2 = list()
    for pred_box, named_pred, pred_score in zip(pred_boxes, named_idx_pairs, pred_scores):
        pred_box = pred_box.int()
        b1 = pred_box[:4]
        b2 = pred_box[4:8]

        n1 = None
        n2 = None

        for box in boxes:
            box = box.int()
            eq_1 = torch.eq(b1, box)
            eq_2 = torch.eq(b2, box)
            eq_all_1 = torch.all(eq_1)
            eq_all_2 = torch.all(eq_2)
            if eq_all_1:
                n1 = box
                m = torch.max(pred_score[1:50])
                index = (pred_score == m).nonzero().flatten()
                p1 = predicates[index]
            if eq_all_2:
                n2 = box
                m = torch.max(pred_score[1:50])
                index = (pred_score == m).nonzero().flatten()
                p2 = predicates[index]

        if n1 is not None and n2 is not None:
            calls += 1
            h1 = hash_tensor(n1)
            h2 = hash_tensor(n2)
            key_1 = (h1,h2)
            key_2 = (h2,h1)
            if key_1 not in pred_score_dict:
                G_in.add_edge(h1, h2)
                l_1.append(key_1)
                n +=1 
                edge_labeldict[key_1] = "{}: {}".format(p1,round(score,2))
                pred_score_dict[key_1] = score
            elif score > pred_score_dict[key_1]:
                edge_labeldict.pop(key_1)
                edge_labeldict[key_1] = "{}: {}".format(p1,round(score,2))
                pred_score_dict[key_1] = score
            else:
                print(key_1)

            if key_2 not in pred_score_dict:
                G_out.add_edge(h2, h1)
                l_2.append(key_2)
                n +=1 
                edge_labeldict[key_2] = "{}: {}".format(p2,round(score,2))
                pred_score_dict[key_2] = score
            elif score > pred_score_dict[key_2]:
                edge_labeldict.pop(key_2)
                edge_labeldict[key_2] = "{}: {}".format(p2,round(score,2))
                pred_score_dict[key_2] = score
            else:
                print(key_2)
    return G_out, labeldict, edge_labeldict

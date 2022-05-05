import cv2
import torch
import json
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
def select_top_predictions(predictions, confidence_threshold=0.2):
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

def build_graph(img_size, prediction, prediction_pred, labels, predicates):
    G = nx.DiGraph()
    labeldict = {}
    edge_labeldict = {}

    image_width = img_size[0]
    image_height = img_size[1]
    prediction = prediction.resize((image_width, image_height))
    obj_scores = prediction.get_field("scores").numpy()
    all_rels = prediction_pred.get_field("idx_pairs").numpy()
    fp_pred = prediction_pred.get_field("scores").numpy()

    scores = np.column_stack((
        obj_scores[all_rels[:,0]],
        obj_scores[all_rels[:,1]],
        fp_pred[:, 1:].max(1)
    )).prod(1)
    sorted_inds = np.argsort(-scores)
    sorted_inds = sorted_inds[scores[sorted_inds] > 0] #[:100]

    pred_entry = {
        'pred_boxes': prediction.bbox.numpy(),
        'pred_classes': prediction.get_field("labels").numpy(),
        'obj_scores': prediction.get_field("scores").numpy(),
        'pred_rel_inds': all_rels[sorted_inds], #object to subject index. List is a realtionship for each score. In each list element the index of subject and object in pred_boxes, pred_classesoir obj_scores is saved
        'rel_scores': fp_pred[sorted_inds], #Score for each possible relation label
    }

    for ind_pair, rel_score in zip(pred_entry['pred_rel_inds'], pred_entry['rel_scores']):
        ind_subject = ind_pair[0]
        ind_object = ind_pair[1]
        subject_label_ind = pred_entry['pred_classes'][ind_subject]
        object_label_ind = pred_entry['pred_classes'][ind_object]
        subject_box = pred_entry['pred_boxes'][ind_subject]
        object_box = pred_entry['pred_boxes'][ind_object]
        subject_score = pred_entry['obj_scores'][ind_subject]
        object_score = pred_entry['obj_scores'][ind_object]
        rel_ind = np.argmax(rel_score)
        subject_label = labels[subject_label_ind]
        object_label = labels[object_label_ind]
        this_rel_score = rel_score[rel_ind]
        this_rel_label = predicates[rel_ind]
        rel_labels_scores = list()
        for i, score in enumerate(rel_score):
            rel_labels_scores.append({'concept': predicates[i], 'score': float(score)})
        # rel_labels_scores_json = json.dumps(rel_labels_scores)
        subject_tuple = box_to_tuple(subject_box)
        object_tuple = box_to_tuple(object_box)

        if object_score > 0.6 and subject_score > 0.6:
            G.add_node(subject_tuple)
            G.add_node(object_tuple)
            labeldict[subject_tuple] = {'concept': subject_label, 'score':float(round(subject_score,2))}
            labeldict[object_tuple] = {'concept': object_label, 'score':float(round(object_score,2))}
            # labeldict[subject_tuple] = "{}: {}".format(subject_label,round(subject_score,2))
            # labeldict[object_tuple] = "{}: {}".format(object_label,round(object_score,2))

            G.add_edge(subject_tuple, object_tuple)
            # edge_labeldict[(subject_tuple, object_tuple)] = "{}: {}".format(this_rel_label,round(this_rel_score,2))
            edge_labeldict[(subject_tuple, object_tuple)] = rel_labels_scores

    return G, labeldict, edge_labeldict

def box_to_tuple(box):
    return (float(box[0]), float(box[1]), float(box[2]), float(box[0]))


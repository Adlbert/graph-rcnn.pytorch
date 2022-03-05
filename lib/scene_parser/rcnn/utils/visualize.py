import cv2
import torch
import numpy as np
import networkx as nx


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
    scores = torch.tensor([torch.mean(score) for score in scores])
    den = torch.max(scores)-torch.min(scores)
    if den == 0:
        den = 1
    scores = (scores - torch.min(scores))/den
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx[0]]

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


def overlay_pred_names(image, predictions, categories):
    scores = predictions.get_field("scores").tolist()
    idx_pairs = predictions.get_field("idx_pairs").tolist()
    idx_pairs = [[categories[j] for j in i] for i in idx_pairs]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, prediction in zip(boxes, scores, idx_pairs):
        w1, h1 = np.abs(box[:2] - box[2:4])
        x1, y1 = (box[0] + w1/2, box[1] + h1/2)
        x1 = round(x1.item())
        y1 = round(y1.item())

        w2, h2 = np.abs(box[4:6] - box[6:8])
        x2, y2 = (box[4] + w2/2, box[5] + h2/2)
        x2 = round(x2.item())
        y2 = round(y2.item())
        s1 = template.format(prediction[0], np.average(score))
        s2 = template.format(prediction[1], np.average(score))
        cv2.putText(
            image, s1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )
        cv2.putText(
            image, s2, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
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
    labels = [categories[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        x = round(x.item())
        y = round(y.item()+20)
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
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

def generate_graph(predictions, pred_predictions, categories, predicates):
    G = nx.Graph()
    labeldict = {}
    edge_labeldict = {}

    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions.bbox

    # for box, score, label in zip(boxes, scores, labels):
    #     G.add_node(str(box))
        # labeldict[str(box)] = label


    pred_scores = pred_predictions.get_field("scores").tolist()
    idx_pairs = pred_predictions.get_field("idx_pairs").tolist()
    idx_pairs = [[predicates[j] for j in i] for i in idx_pairs]
    pred_boxes = predictions.bbox

    for pred_box, pred_score, prediction, box, score, label in zip(pred_boxes, pred_scores, idx_pairs, boxes, scores, labels):
        b1 = pred_box[:4]
        b2 = pred_box[5:8]
        labeldict[b1] = prediction[0]
        labeldict[b2] = prediction[1]
        edge_labeldict[box] = "{}: {}".format(label,score)
        G.add_edge(b1, b2)


    return G, labeldict, edge_labeldict

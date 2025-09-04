import supervision as sv

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.default(), 
    thickness=2, 
    text_thickness=1, 
    text_scale=0.5,
    text_padding=2
)

def draw_detections(image, boxes, scores, class_ids, class_map):

    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids,
        tracker_id=None
    )

    labels = [
        f"{class_map[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _ in detections
    ]
    
    out_image = BOX_ANNOTATOR.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels,
        skip_label=False
    )

    return out_image
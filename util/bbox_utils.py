
# Returns the center of the bounding box
def get_center_of_bbox(bbox):

    # Locations
    x1, y1, x2, y2 = bbox

    # Returning the center
    return (int((x1+x2) / 2), int((y1 + y2) / 2))

# Returns the width of a bounding box
def get_bbox_width(bbox):
    return bbox[2]-bbox[0]
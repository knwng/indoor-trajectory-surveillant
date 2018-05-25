class FrameInst(object):
    '''
        Data structure for detected instance
    '''
    def __init__(self, cropped_img_url, timestamp, bbox, identity=None, embedding=None):
        self.cropped_img_url = cropped_img_url
        self.timestamp = timestamp
        # [xmin, xmax, ymin, ymax]
        self.bbox = bbox
        self.identity = identity
        self.embedding = embedding

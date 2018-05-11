class DetectedInst(object):
    '''
        Data structure for detected instance
    '''

    def __init__(self, cropped_img, timestamp, boundingbox, identity):
        self.cropped_img = cropped_img
        self.timestamp = timestamp
        self.boundingbox = boundingbox
        self.identity = identity


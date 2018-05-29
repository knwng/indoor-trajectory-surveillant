class FrameInst(object):
    '''
        Data structure for detected instance
        cropped_img_url: string | url of cropped and resized (256 x 128) image
        timestamp: int | timestamp of current frame (from the beginning of video)
        bbox: 4-d array [xmin, xmax, ymin, ymax] | bbox of person in original surveillant video
        identity: int | id of person, -1 for unrecognized person
        id_candidates: n-d array [id, score] | indicating the id candidate and its score
        embedding: 128-d array | embedding of person, used for re-id
    '''
    def __init__(self, cropped_img_url, timestamp, bbox, identity=None, id_candidates=None, embedding=None):
        self.cropped_img_url = cropped_img_url
        self.timestamp = timestamp
        # [xmin, xmax, ymin, ymax]
        self.bbox = bbox
        self.identity = identity
        self.id_candidates = id_candidates
        self.embedding = embedding

    def display(self):
        print('url {}\ntimestamp {}\nbbox {}\nidentity {}\nidc {}\nembedding {}'.format(
                        self.cropped_img_url,
                        self.timestamp,
                        self.bbox,
                        self.identity,
                        self.id_candidates,
                        self.embedding))

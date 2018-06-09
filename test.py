#!/usr/bin/env python3

from surveillant import *


def TEST_JOINT_DET_REID():
    DETECT_INTERVAL = 4
    STRUCT_STORAGE_ROOT = './output/struct_storage'
    RAW_STORAGE_ROOT = './output/raw'

    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    video_dir = './videos/0509_1_A3_ELEVATOR_slice.mp4'
    # output_root_dir = os.path.basename(video_dir).split('.')[0]
    STORAGE_BASENAME = os.path.basename(video_dir).split('.')[0]
    if not os.path.exists(STRUCT_STORAGE_ROOT):
        os.makedirs(STRUCT_STORAGE_ROOT)
    if not os.path.exists(RAW_STORAGE_ROOT):
        os.makedirs(RAW_STORAGE_ROOT)
    # if not os.path.exists(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)):
    #     os.makedirs(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME))

    cap = cv2.VideoCapture(video_dir)
    frame_idx = 0
    struct_info_storage = {'video_fn':video_dir, 
                           'frames':[], 
                           'timestamp_offset':0.0,
                           'detect_interval':self.detect_interval,
                           'storage_dir':os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)}
    raw_det_storage = {}
    while cap.isOpened():
        frame_idx += 1
        if frame_idx % DETECT_INTERVAL != 0:
            continue
        ret, frame = cap.read()
        if frame is None:
            print('No frame stream')
            break
        height, width, _ = frame.shape
        rclasses, rscores, rbboxes = surve.detector.detect(frame, viz=False)
        if len(rbboxes) == 0:
            continue
        identity_idx = 1
        cropped_imgs = [cv2.resize(frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :], (128, 256)) for bbox in rbboxes]
        embs = surve.reid.embed(cropped_imgs)
        raw_det_storage['frame-{}'.format(frame_idx)] = cropped_imgs

        for _bbox, _emb in zip(rbboxes, embs):
            # cropped_img_url = os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME, 'frame-{}-idx-{}.jpg'.format(frame_idx, identity_idx))
            cropped_img_url=''
            struct_info_storage['frames'].append(FrameInst(cropped_img_url=cropped_img_url,
                                                           timestamp=frame_idx,
                                                           bbox=_bbox,
                                                           embedding=_emb))

            identity_idx += 1
        print('preparing | frame {} | id {}'.format(frame_idx, identity_idx-1))
    with open(os.path.join(STRUCT_STORAGE_ROOT, STORAGE_BASENAME)+'.pkl', 'wb') as f:
        pickle.dump(struct_info_storage, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)+'.pkl', 'wb') as f:
        pickle.dump(raw_det_storage, f, pickle.HIGHEST_PROTOCOL)
    return

def TEST_REID():
    TOPK = 5
    GALLERY_ROOT = 'gallery_set_full'
    QUERY_ROOT = 'query_set'
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    gallery_fn_list = [os.path.join(GALLERY_ROOT, x) for x in os.listdir(GALLERY_ROOT) if x.endswith('.jpg')]
    query_fn_set = [os.path.join(QUERY_ROOT, x) for x in os.listdir(QUERY_ROOT) if x.endswith('.jpg')]
    print(gallery_fn_list)
    print(query_fn_set)
    print('load embedding into gallery')
    start = time.time()
    surve.load_gallery(gallery_fn_list)
    print('gallery loading time {} per image'.format((time.time() - start) / len(gallery_fn_list)))
    for query in query_fn_set:
        query_img = cv2.imread(query)
        surve.query(query_img, viz=True)
    cv2.waitKey()
    return

def TEST_EXTRACT_FEATURE():
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000', detect_interval=1, reid_threshold=0.09, reid_register_threshold=0.15)
    img_list = os.listdir('./reid_test_data')
    _stor = []
    # in format [seq_idx, id_idx, img_idx, ftrs]
    for img_fn in img_list:
        _, seq_idx, _, id_idx, img_idx = img_fn.split('.')[0].split('_')
        img = cv2.imread(os.path.join('./reid_test_data', img_fn))
        if img is None:
            print('{} not exist'.format(img_fn))
            continue
        embs = surve.reid.embed(img, augmentation=True, pre_crop_size=[144, 288])[0]
        print('embs {}'.format(np.array(embs).shape))
        _stor.append([seq_idx, id_idx, img_idx, embs])
    with open('reid_ftr_augmented_storage.pkl', 'wb') as f:
        pickle.dump(_stor, f, pickle.HIGHEST_PROTOCOL)
    return

def TEST_PREPARE_ID():
    DETECT_INTERVAL = 4 
    # video_dir = './videos/0509_1_2_PASS_slice.mp4'
    video_dir = './videos/0509_1_A3_ELEVATOR_slice.mp4'
    output_root_dir = os.path.basename(video_dir)
    if not os.path.exists(output_root_dir):
        os.mkdir(output_root_dir)

    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    cap = cv2.VideoCapture(video_dir)
    frame_idx = 0
    while cap.isOpened():
        frame_idx += 1
        if frame_idx % DETECT_INTERVAL != 0:
            continue
        ret, frame = cap.read()
        if frame is None:
            print('No frame stream')
            break
        height, width, _ = frame.shape
        rclasses, rscores, rbboxes = surve.detector.detect(frame, viz=False)
        if len(rbboxes) == 0:
            continue
        identity_idx = 1
        for bbox in rbboxes:
            id_img = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
            id_img = cv2.resize(id_img, (128, 256))
            cv2.imwrite(os.path.join(output_root_dir, 'frame-{}-id-{}.jpg'.format(frame_idx, identity_idx)), id_img)
            identity_idx += 1
        print('preparing | frame {} | id {}'.format(frame_idx, identity_idx-1))

def TEST_DET():
    img = cv2.imread("./test.jpg")
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    surve.detector.detect(img, viz=True)
    cv2.waitKey()


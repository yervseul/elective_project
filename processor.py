import cv2
import numpy as np
import os
import mediapipe as mp

mp_detect = mp.solutions.face_detection
MAX_FACES = 12


# -----------------------------
# NON MAX SUPPRESSION
# -----------------------------
def nms(boxes, overlapThresh=0.4):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:

        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)

        overlap = (w*h)/area[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs)-1],
            np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype(int)


# -----------------------------
# CROSS MASK
# -----------------------------
def cross_mask(shape, cx, cy, size):

    h,w = shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)

    thickness = int(size*0.18)
    arm = int(size*0.8)

    cv2.rectangle(mask,(cx-thickness,cy-arm*2),(cx+thickness,cy+arm),255,-1)
    cv2.rectangle(mask,(cx-arm,cy-thickness),(cx+arm,cy+thickness),255,-1)

    mask = cv2.GaussianBlur(mask,(41,41),0)
    return mask


# -----------------------------
# GRAIN
# -----------------------------
def add_grain(img,strength=18):

    noise = np.random.normal(0,strength,img.shape).astype(np.int16)
    out = img.astype(np.int16)+noise
    return np.clip(out,0,255).astype(np.uint8)


# -----------------------------
# STATIC LINES
# -----------------------------
def add_static(img):

    out = img.copy()
    h,w = out.shape[:2]

    for y in range(0,h,4):
        out[y,:,:]=(out[y,:,:]*0.92).astype(np.uint8)

    return out


# -----------------------------
# DETECT FACES ON ONE IMAGE
# -----------------------------
def detect_once(img):

    h,w = img.shape[:2]
    boxes = []

    with mp_detect.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as detector:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        if res.detections:

            for d in res.detections:

                b = d.location_data.relative_bounding_box

                x1 = int(b.xmin*w)
                y1 = int(b.ymin*h)
                x2 = int((b.xmin+b.width)*w)
                y2 = int((b.ymin+b.height)*h)

                boxes.append([x1,y1,x2,y2])

    return boxes


# -----------------------------
# ROTATION DETECTION
# -----------------------------
def detect_faces_all_angles(img):

    h,w = img.shape[:2]
    all_boxes = []

    rotations = [
        (img,0),
        (cv2.rotate(img, cv2.ROTATE_180),180)
    ]

    for rotated,angle in rotations:

        boxes = detect_once(rotated)

        for x1,y1,x2,y2 in boxes:

            if angle == 0:
                rx1,ry1,rx2,ry2 = x1,y1,x2,y2

            else: # 180°
                rx1 = w-x2
                ry1 = h-y2
                rx2 = w-x1
                ry2 = h-y1

            all_boxes.append([rx1,ry1,rx2,ry2])

    # remove duplicates
    return nms(all_boxes)


# -----------------------------
# EFFECT
# -----------------------------
def apply_effect(img):

    boxes = detect_faces_all_angles(img)

    faces = []
    for (x1,y1,x2,y2) in boxes[:MAX_FACES]:

        cx = (x1+x2)//2
        cy = (y1+y2)//2
        size = int((y2-y1)*0.45)

        faces.append((cx,cy,size))

    for cx,cy,size in faces:

        mask = cross_mask(img.shape,cx,cy,size)
        inverted = cv2.bitwise_not(img)

        alpha = mask/255.0
        alpha = np.stack([alpha]*3,axis=2)

        img = (img*(1-alpha) + inverted*alpha).astype(np.uint8)

    img = add_grain(img,18)
    img = add_static(img)

    return img


# -----------------------------
# BATCH
# -----------------------------
def process_all_images(input_path, output_path, processed_path="processed"):

    os.makedirs(output_path,exist_ok=True)
    os.makedirs(processed_path,exist_ok=True)

    for file in os.listdir(input_path):

        if not file.lower().endswith((".png",".jpg",".jpeg")):
            continue

        path = os.path.join(input_path,file)
        img = cv2.imread(path)

        if img is None:
            continue

        out = apply_effect(img)

        cv2.imwrite(os.path.join(output_path,"cross_"+file),out)
        os.replace(path, os.path.join(processed_path,file))

        print("processed:",file)

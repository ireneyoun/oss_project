#---- Define processing functions -------#
import sys
import torch
import cv2
import numpy as np
import torchvision
import onnxruntime as ort
import time, random
import smtplib
from email.mime.text import MIMEText

img_size = 640

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(prediction, conf_thres, iou_thres, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nm=0):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    mps = 'mps' in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            x = x[x[:, 4].argsort(descending=True)]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            break

    return output

def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def class_name():
    classes=['noraml', 'abnormal']
    return classes

def letterbox(im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    new_shape= img_size
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def ort_session(onnx_model):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model, providers=providers)
    print(session.get_providers())
    return session

def send_alert_email():
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login('sunyoungju517@gmail.com', 'scfz vnwo qpud vrbv')

    msg = MIMEText('내용 : 이상 행동 감지')
    msg['Subject'] = '제목: 이상 행동 감지'

    smtp.sendmail('sunyoungju517@gmail.com', 'sts07190@naver.com', msg.as_string())
    smtp.quit()

def result(img,ratio, dwdh, out):
    names= class_name()
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    abnormal_detected = False

    for i,(x0,y0,x1,y1,score,cls_id) in enumerate(out):
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(img,box[:2],box[2:],color,2)
        cv2.putText(img,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        
        if name.startswith("abnormal"):
            abnormal_detected = True

    if abnormal_detected:
        send_alert_email()

    return img

cuda = True
w = "best.onnx"

webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

cnt = 0
while webcam.isOpened():
    status, img = webcam.read()
    if not status:
        print("Could not read frame")
        break

    image, ratio, dwdh = letterbox(img, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255

    session= ort_session(w)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]

    inp = {inname[0]:im}

    t1 = time.time()
    outputs = session.run(outname, inp)[0]
    t2 = time.time()
    output = torch.from_numpy(outputs)
    out = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)[0]
    consumed_time = t2-t1
    print('The {}-th frame yolov5 ONNXRuntime Inference Time: {}(msec)'.format(cnt, consumed_time))
    imgout = result(img, ratio, dwdh, out)
    cv2.imwrite('detection-result.jpg', imgout)
    cv2.imshow('detection-result', imgout)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    cnt = cnt + 1

webcam.release()
cv2.destroyAllWindows()

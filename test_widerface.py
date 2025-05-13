import argparse
import glob
import time
from pathlib import Path

import os
import cv2
import torch
import shutil
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression_face, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm

def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride 
    return max_size

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def detect(model, img0):
    stride = int(model.stride.max())  # model stride
    imgsz = opt.img_size
    if imgsz <= 0:                    # original size    
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    output = model(img, augment=opt.augment)
    
    # Adaptation pour PyTorch 2.6+ : Gestion des différents formats de sortie possibles
    if isinstance(output, tuple):
        # Si la sortie est un tuple, le premier élément contient généralement les prédictions
        pred = output[0]
    elif isinstance(output, list):
        # Si la sortie est une liste
        pred = output[0]  
    elif isinstance(output, torch.Tensor):
        # Si la sortie est directement un tensor
        pred = output
    else:
        # Fallback si la structure est différente
        pred = output
    
    # Apply NMS
    pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
    boxes = []
    h, w, c = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred[:, 5:15] = scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = pred[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2-x1, y2-y1, conf])
    return boxes


if __name__ == '__main__':
    # Créer un dossier de debug
    debug_dir = './debug_output'
    os.makedirs(debug_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='../WiderFace/val/images/', type=str, help='dataset path')
    parser.add_argument('--folder_pict', default='/yolov5-face/data/widerface/val/wider_val.txt', type=str, help='folder_pict')
    opt = parser.parse_args()
    print(opt)

    # changhy : read folder_pict
    pict_folder = {}
    with open(opt.folder_pict, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('\\', '/').split('/')
            pict_folder[line[-1]] = line[-2]
    
    # Debug: afficher quelques entrées du dictionnaire pict_folder
    print(f"Nombre d'images dans pict_folder: {len(pict_folder)}")
    sample_keys = list(pict_folder.keys())[:5]
    print(f"Exemples d'entrées dans pict_folder:")
    for key in sample_keys:
        print(f"  {key} -> {pict_folder[key]}")

    # Load model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    print(f"Modèle chargé: {opt.weights}")
    with torch.no_grad():
        # testing dataset
        testset_folder = opt.dataset_folder

        # Créer un fichier de log pour les détections
        with open('./debug_output/detection_log.txt', 'w') as log_file:
            log_file.write(f"Dataset folder: {testset_folder}\n")
            log_file.write(f"Save folder: {opt.save_folder}\n")
            log_file.write(f"Nombre d'événements attendus: {len(set(pict_folder.values()))}\n")
            log_file.write(f"Événements attendus: {set(pict_folder.values())}\n\n")
            
            image_count = 0
            detection_count = 0
            
            # Créer les dossiers pour tous les événements à l'avance
            for event_name in set(pict_folder.values()):
                event_dir = os.path.join(opt.save_folder, event_name)
                os.makedirs(event_dir, exist_ok=True)
                
            for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*', '*'))):
                if image_path.endswith('.txt'):
                    continue
                image_count += 1
                img0 = cv2.imread(image_path)  # BGR
                if img0 is None:
                    log_file.write(f'ignore : {image_path}\n')
                    print(f'ignore : {image_path}')
                    continue
                    
                # Sauvegarder quelques images originales pour débogage
                if image_count < 10:
                    debug_img_path = os.path.join(debug_dir, f"original_{image_count}.jpg")
                    cv2.imwrite(debug_img_path, img0)
                    
                boxes = detect(model, img0)
                detection_count += len(boxes)
                
                # Log pour débogage
                image_name = os.path.basename(image_path)
                log_file.write(f"Image: {image_path} - Détections: {len(boxes)}\n")
                # --------------------------------------------------------------------
                image_name = os.path.basename(image_path)
                
                # Vérifier si l'image est dans le dictionnaire pict_folder
                if image_name not in pict_folder:
                    log_file.write(f"ERREUR: Image {image_name} non trouvée dans pict_folder!\n")
                    parts = image_path.replace('\\', '/').split('/')
                    if len(parts) >= 2:
                        event_name = parts[-2]  # Le dossier parent de l'image est l'événement
                        log_file.write(f"  Utilisation de l'événement extrait du chemin: {event_name}\n")
                        pict_folder[image_name] = event_name
                    else:
                        log_file.write(f"  Impossible de déterminer l'événement pour {image_name}\n")
                        continue
                
                # Sauvegarder une image avec détections pour débogage
                if image_count < 10:
                    debug_img = img0.copy()
                    for box in boxes:
                        cv2.rectangle(debug_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                    debug_img_path = os.path.join(debug_dir, f"detection_{image_count}.jpg")
                    cv2.imwrite(debug_img_path, debug_img)
                
                txt_name = os.path.splitext(image_name)[0] + ".txt"
                save_name = os.path.join(opt.save_folder, pict_folder[image_name], txt_name)
                dirname = os.path.dirname(save_name)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                    log_file.write(f"Création du dossier: {dirname}\n")
                
                with open(save_name, "w") as fd:
                    file_name = os.path.basename(save_name)[:-4] + "\n"            
                    bboxs_num = str(len(boxes)) + "\n"
                    fd.write(file_name)
                    fd.write(bboxs_num)
                    for box in boxes:
                        fd.write('%d %d %d %d %.03f' % (box[0], box[1], box[2], box[3], box[4] if box[4] <= 1 else 1) + '\n')
                        
                # Log les 5 premières sauvegardes pour vérifier le format
                if image_count < 5:
                    log_file.write(f"  Sauvegarde dans: {save_name}\n")
            log_file.write(f"\nStatistiques finales:\n")
            log_file.write(f"  Total d'images traitées: {image_count}\n")
            log_file.write(f"  Total de détections: {detection_count}\n")
            
            # Liste les fichiers dans le répertoire de sortie pour vérification
            log_file.write(f"\nContenu du répertoire de sortie {opt.save_folder}:\n")
            events_created = os.listdir(opt.save_folder)
            log_file.write(f"  Événements créés ({len(events_created)}): {events_created}\n")
            
            for event in events_created:
                event_path = os.path.join(opt.save_folder, event)
                if os.path.isdir(event_path):
                    files = os.listdir(event_path)
                    log_file.write(f"  Nombre de fichiers dans {event}: {len(files)}\n")
                    
            # Copier le journal de débogage dans le dossier d'évaluation pour référence
            shutil.copy('./debug_output/detection_log.txt', os.path.join(opt.save_folder, 'detection_log.txt'))
            
        print('done.')

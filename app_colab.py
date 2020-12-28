#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import time
import numpy as np
import pandas as pd
import threading
#from yolov4.tf import YOLOv4
import matplotlib.pyplot as plt
from collections import defaultdict


# In[2]:


from deep_sort.tools.generate_detections import create_box_encoder
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker as ds_Tracker
MODEL_CKPT = "./deep_sort/weights/mars-small128.pb"
import action_detection.action_detector as act
COLORS = np.random.randint(0, 255, [1000, 3])


# In[3]:


class Tracker():
    def __init__(self, timesteps=32):
        self.active_actors = []
        self.inactive_actors = []
        self.actor_no = 0
        self.frame_history = []
        self.frame_no = 0
        self.timesteps = timesteps
        self.actor_infos = {}
        # deep sort
        self.encoder = create_box_encoder(MODEL_CKPT, batch_size=16)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None) #, max_cosine_distance=0.2) #, nn_budget=None)
        self.tracker = ds_Tracker(metric, max_iou_distance=0.7, max_age=200, n_init=5)
        self.score_th = 0.40

    def update_tracker(self, detection_info, frame):
        ''' Takes the frame and the results from the object detection
            Updates the tracker wwith the current detections and creates new tracks
        '''

        boxes = np.array([d[:4] for d in detection_info])
        scores = np.array([d[4] for d in detection_info])
        num_detections = len(detection_info)
        indices = scores > self.score_th # filter score threshold
        filtered_boxes, filtered_scores = boxes[indices], scores[indices]

        H,W,C = frame.shape
        filtered_boxes[:, [0, 2]] = filtered_boxes[:, [0, 2]] * W
        filtered_boxes[:, [1, 3]] = filtered_boxes[:, [1, 3]] * H
        # deep sort format boxes (x, y, W, H)
        ds_boxes = []
        for bb in range(filtered_boxes.shape[0]):
            cur_box = filtered_boxes[bb]
            cur_score = filtered_scores[bb]
            c_x = int(cur_box[0])
            c_y = int(cur_box[1])
            half_w = int(cur_box[2]/2)
            half_h = int(cur_box[3]/2)
            ds_box = [c_x - half_w, c_y - half_h, int(cur_box[2]), int(cur_box[3])]
            ds_boxes.append(ds_box)
        features = self.encoder(frame, ds_boxes)

        detection_list = []
        for bb in range(filtered_boxes.shape[0]):
            cur_box = filtered_boxes[bb]
            cur_score = filtered_scores[bb]
            feature = features[bb]
            c_x = int(cur_box[0])
            c_y = int(cur_box[1])
            half_w = int(cur_box[2]/2)
            half_h = int(cur_box[3]/2)
            ds_box = [c_x - half_w, c_y - half_h, int(cur_box[2]), int(cur_box[3])]
            detection_list.append(Detection(ds_box, cur_score, feature))

        # update tracker
        self.tracker.predict()
        self.tracker.update(detection_list)
        
        # Store results.
        actives = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            left, top, width, height = bbox
            tr_box = [top / float(H), left / float(W), (top+height)/float(H), (left+width)/float(W)]
            actor_id = track.track_id
            detection_conf = track.last_detection_confidence
            
            if actor_id in self.actor_infos: # update with the new bbox info
                cur_actor = self.actor_infos[actor_id]
                no_interpolate_frames = self.frame_no - cur_actor['last_updated_frame_no']
                interpolated_box_list = bbox_interpolate(cur_actor['all_boxes'][-1], tr_box, no_interpolate_frames)
                cur_actor['all_boxes'].extend(interpolated_box_list[1:])
                cur_actor['last_updated_frame_no'] = self.frame_no
                cur_actor['length'] = len(cur_actor['all_boxes'])
                cur_actor['all_scores'].append(detection_conf)
                actives.append(cur_actor)
            else:
                new_actor = {'all_boxes': [tr_box], 'length':1, 'last_updated_frame_no': self.frame_no, 'all_scores':[detection_conf], 'actor_id':actor_id}
                self.actor_infos[actor_id] = new_actor

        self.active_actors = actives
        
        self.frame_history.append(frame)
        if len(self.frame_history) > 2*self.timesteps:
            del self.frame_history[0]

        self.frame_no += 1

    def generate_all_rois(self):
        no_actors = len(self.active_actors)
        rois_np = np.zeros([no_actors, 4])
        temporal_rois_np = np.zeros([no_actors, self.timesteps, 4])
        for bb, actor_info in enumerate(self.active_actors):
            actor_no = actor_info['actor_id']
            norm_roi, full_roi = self.generate_person_tube_roi(actor_no)
            rois_np[bb] = norm_roi
            temporal_rois_np[bb] = full_roi
        return rois_np, temporal_rois_np

    def generate_person_tube_roi(self, actor_id):
        actor_info = [act for act in self.active_actors if act['actor_id'] == actor_id][0]
        boxes = actor_info['all_boxes']
        if actor_info['length'] < self.timesteps:
            recent_boxes = boxes
            index_offset = (self.timesteps - actor_info['length'] + 1) 
        else:
            recent_boxes = boxes[-self.timesteps:]
            index_offset = 0
        H,W,C = self.frame_history[-1].shape
        mid_box = recent_boxes[len(recent_boxes)//2]
        edge, norm_roi = generate_edge_and_normalized_roi(mid_box)

        full_rois = []
        for rr in range(self.timesteps):
            if rr < index_offset:
                cur_box = recent_boxes[0]
            else:
                cur_box = recent_boxes[rr - index_offset]

            top, left, bottom, right = cur_box
            cur_center = (top+bottom)/2., (left+right)/2.
            top, bottom = cur_center[0] - edge, cur_center[0] + edge
            left, right = cur_center[1] - edge, cur_center[1] + edge
            
            full_rois.append([top, left, bottom, right])
        full_rois_np = np.stack(full_rois, axis=0)

        return norm_roi, full_rois_np


# In[4]:


def bbox_interpolate(start_box, end_box, no_interpolate_frames):
    delta = (np.array(end_box) - np.array(start_box)) / float(no_interpolate_frames)
    interpolated_boxes = []
    for ii in range(0, no_interpolate_frames+1):
        cur_box = np.array(start_box) + delta * ii
        interpolated_boxes.append(cur_box.tolist())
    return interpolated_boxes


# In[5]:


def generate_edge_and_normalized_roi(mid_box):
    top, left, bottom, right = mid_box

    edge = max(bottom - top, right - left) / 2. * 1.5 # change this to change the size of the tube

    cur_center = (top+bottom)/2., (left+right)/2.
    context_top, context_bottom = cur_center[0] - edge, cur_center[0] + edge
    context_left, context_right = cur_center[1] - edge, cur_center[1] + edge

    normalized_top = (top - context_top) / (2*edge)
    normalized_bottom = (bottom - context_top) / (2*edge)

    normalized_left = (left - context_left) / (2*edge)
    normalized_right = (right - context_left) / (2*edge)

    norm_roi = [normalized_top, normalized_left, normalized_bottom, normalized_right]

    return edge, norm_roi


# In[6]:


def buildActionDict():
    with open("ava_videos/action_list.pbtxt", 'r') as file:
        actions = file.read()
    actions = actions.split('item {\n  ')[1:]
    actions = [[keys.split(': ') for keys in ac.split('\n')[:2]] for ac in actions]
    actions_dict ={}
    for ac in actions:
        actions_dict[int(ac[1][1])] = ac[0][1][1:-1]
    return actions_dict


# In[7]:


def getGroundTruthBbox(df):
    bboxes = []
    for idx, row in df.iterrows():
        bboxes.append([row['x1'], row['y1'], row['x2'], row['y2'], row['action_id']])
    return np.array(bboxes)


# In[8]:


def draw_groundTruth_bboxes(image, bboxes):
    image = np.copy(image)
    height, width, _ = image.shape
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
    actions = buildActionDict()
    for bbox in bboxes:
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        action_id = bbox[4]
        bbox_color = (255, 0, 255)
        font_size = 0.4
        font_thickness = 1
        cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)
        bbox_text = actions[action_id]
        t_size = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        cv2.rectangle(
            image,
            top_left,
            (top_left[0] + t_size[0], top_left[1] - t_size[1] - 3),
            bbox_color,
            -1,
        )
        cv2.putText(
            image,
            bbox_text,
            (top_left[0], top_left[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255 - bbox_color[0], 255 - bbox_color[1], 255 - bbox_color[2]),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


# In[9]:


def draw_objects(image, bboxes, classes):
    image = np.copy(image)
    height, width, _ = image.shape
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
    person_count = 0
    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)
        top_left = [c_x - half_w, c_y - half_h]
        bottom_right = [c_x + half_w, c_y + half_h]
        top_left[0] = max(top_left[0], 0)
        top_left[1] = max(top_left[1], 0)
        bottom_right[0] = min(bottom_right[0], width)
        bottom_right[1] = min(bottom_right[1], height)
        class_id = int(bbox[4])
        if class_id == 0:
            person_count += 1
            windowName = "{}_{}".format(classes[class_id],person_count)
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            obj = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            cv2.imshow(windowName, obj)


# In[10]:


def buildYoloModel():
    yolo = YOLOv4()
    yolo.classes = "coco.names"
    yolo.input_size=(608,608)
    yolo.make_model()
    yolo.load_weights("yolov4.weights", weights_type='yolo')
    return yolo


# In[11]:


def objectDetection(path, media_name, yolo, iou_threshold = 0.5, score_threshold = 0.5, start_time = 902, end_time = 1798):
    
    media_path = path + media_name
    
    if not os.path.exists(media_path):
        raise FileNotFoundError("{} does not exist".format(media_path))

    cap = cv2.VideoCapture(media_path)

    if cap.isOpened():
        while True:
            try:
                is_success, frame = cap.read()
            except cv2.error:
                continue
                
            now_second = cap.get(0)/1000
            
            if now_second < start_time: continue
            if (not is_success) or (now_second >= end_time+1): break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            bboxes = yolo.predict(
                frame,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            bboxes.view('i8,i8,i8,i8,i8,i8').sort(order=['f0','f1'], axis=0)
            for bb in bboxes:
                if bb[4] == 0:
                    obj = [media_name, now_second]+list(bb)
                    objs.append(obj)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()


# In[12]:


def rebuildBBoxes(media_name):
    bbox_df = pd.read_csv('bbox_res/'+media_name+'.csv')
    cur_timestamp = 0
    bboxes = []
    for index, row in bbox_df.iterrows():
        if row['timestamp'] != cur_timestamp:
            bboxes.append([])
            cur_timestamp = row['timestamp']    
        bboxes[-1].append([row['c_x'],row['c_y'],row['w'],row['h'],row['confidence']])
    return bboxes


# In[13]:


def store_detection_results(active_actors, prob_dict, timestamp, media_name):
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']
        cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
        try:
            cur_box, cur_score, cur_class = cur_actor['all_boxes'][-16], cur_actor['all_scores'][0], 1
        except IndexError:
            continue
        act_dict = buildActionDict()
        act_dict = {v: k for k, v in act_dict.items()}
        top, left, bottom, right = cur_box
        for act in cur_act_results:
            res_list.append([media_name, timestamp, left, top, right, bottom, act_dict[act[0]], actor_id])


# In[14]:


def visualize_detection_results(img_np, active_actors, prob_dict):
    
    score_th = 0.30
    action_th = 0.20

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape
    font_thickness = 1
    
    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']
        cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
        try:
            cur_box, cur_score, cur_class = cur_actor['all_boxes'][-16], cur_actor['all_scores'][0], 1
        except IndexError:
            continue
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box

        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score
        label = 'person'
        message = '%s_%i: %% %.2f' % (label, actor_id, conf)
        action_message_list = ["%s:%.3f" % (actres[0], actres[1]) for actres in cur_act_results if actres[1]>action_th]

        color = COLORS[actor_id]
        color = (int(color[0]), int(color[1]), int(color[2]))
        
        cv2.rectangle(disp_img, (left,top), (right,bottom), color, 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(
            disp_img, 
            message, 
            (left, top-12), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_size, 
            (255-color[0],255-color[1],255-color[2]), 
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        #action message writing
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color, -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            cv2.putText(
                disp_img, 
                action_message, 
                (left, top+5+offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255-color[0],255-color[1],255-color[2]), 
                font_thickness,
                lineType=cv2.LINE_AA,
            )

    return disp_img


# In[15]:


def run(path, media_name, iou_threshold = 0.5, score_threshold = 0.5, start_time = 902, end_time = 1798):
    
    print("Processing:", media_name)
    
    media_path = path+media_name

    #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    
    bbox_list = rebuildBBoxes(media_name)

    cap = cv2.VideoCapture(media_path)
    fps = cap.get(5)
    W, H = int(cap.get(3)), int(cap.get(4))
    out_video = cv2.VideoWriter('/content/drive/MyDrive/DataScienceProject/res_video/'+media_name+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (W,H))
    
    tracker = Tracker()
    action_freq = 8
    T = tracker.timesteps
    act_detector = act.Action_Detector('soft_attn')
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'
    memory_size = act_detector.timesteps - action_freq
    updated_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf_with_memory([T,H,W,3], memory_size)
    
    rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)
    
    ckpt_path = '/content/drive/MyDrive/DataScienceProject/'+ckpt_name
    act_detector.restore_model(ckpt_path)

    prob_dict = {}
    
    frame_cnt = 0
    sec_list = []

    if cap.isOpened():
        while True:
            try:
                is_success, frame = cap.read()
            except cv2.error:
                continue
                
            now_second = cap.get(0)/1000
            
            if now_second < start_time: continue
            if (not is_success) or (now_second >= end_time+1): break
                
            bboxes = bbox_list[frame_cnt]
            sec_list.append(now_second)
            frame_cnt += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            tracker.update_tracker(bboxes, frame)
            no_actors = len(tracker.active_actors)
            
            if tracker.active_actors and frame_cnt % action_freq == 0:
                probs = []

                cur_input_sequence = np.expand_dims(np.stack(tracker.frame_history[-action_freq:], axis=0), axis=0)

                rois_np, temporal_rois_np = tracker.generate_all_rois()
                if no_actors > 14:
                    no_actors = 14
                    rois_np = rois_np[:14]
                    temporal_rois_np = temporal_rois_np[:14]

                feed_dict = {updated_frames:cur_input_sequence, # only update last #action_freq frames
                             temporal_rois: temporal_rois_np,
                             temporal_roi_batch_indices: np.zeros(no_actors),
                             rois:rois_np, 
                             roi_batch_indices:np.arange(no_actors)}
                run_dict = {'pred_probs': pred_probs}
                out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)
                probs = out_dict['pred_probs']
                # associate probs with actor ids
                print_top_k = 5
                for bb in range(no_actors):
                    act_probs = probs[bb]
                    order = np.argsort(act_probs)[::-1]
                    cur_actor_id = tracker.active_actors[bb]['actor_id']
                    print("Person %i" % cur_actor_id)
                    cur_results = []
                    for pp in range(print_top_k):
                        print('\t %s: %.3f' % (act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                        cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
                    prob_dict[cur_actor_id] = cur_results

            if frame_cnt > 16:
                store_detection_results(tracker.active_actors, prob_dict, sec_list[-16], media_name)
                out_img = visualize_detection_results(tracker.frame_history[-16], tracker.active_actors, prob_dict)
                #cv2.imshow('result', out_img[:,:,::-1])
                out_video.write(out_img[:,:,::-1])
                cv2.waitKey(10)

                
            #groundTruth_bboxes = getGroundTruthBbox(groundTruth_df[groundTruth_df['timestamp']==int(now_second)])

            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #image = yolo.draw_bboxes(frame, bboxes)
            #groundTruth_img = draw_groundTruth_bboxes(frame, groundTruth_bboxes)

            #cv2.imshow("result", image)
            #cv2.imshow("origin", frame)
            #cv2.imshow("ground_truth", groundTruth_img)
            #draw_objects(frame, bboxes, yolo.classes)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# In[16]:


with open('/content/drive/MyDrive/DataScienceProject/ava_file_names_trainval_v2.1.txt', 'r') as f:
    video_names = f.readlines()
video_names = [v.rstrip().split('.') for v in video_names]
video_names_dict = {}
for video in video_names:
    video_names_dict[video[0]] = video[0]+'.'+video[1]

columns = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']
#train_df = pd.read_csv('ava_videos/ava_train_v2.2.csv')
val_df = pd.read_csv('/content/drive/MyDrive/DataScienceProject/ava_val_v2.2.csv')
#train_df.columns = columns
val_df.columns = columns
#train_df.dropna(inplace=True)
val_df.dropna(inplace=True)
#train_df.drop(train_df[train_df['video_id']=='#NAME?'].index, inplace=True)
#train_df['video_id'] = train_df['video_id'].map(video_names_dict)
val_df['video_id'] = val_df['video_id'].map(video_names_dict)

#train_videos = train_df['video_id'].unique()
val_videos = val_df['video_id'].unique()

#train_path = "ava_videos/train/"
val_path = "https://s3.amazonaws.com/ava-dataset/trainval/"


# In[ ]:


#yolo = buildYoloModel()
#cnt = 0
#for media_name in val_videos:
#    start = time.time()
#    print("Video", cnt, ", Processing:", media_name)
#    objectDetection(val_path, media_name, yolo, start_time = 1546.7452)
#    print("Done in ", time.time()-start, 'seconds')
#    res = pd.DataFrame(objs, columns=['video_id','timestamp','c_x','c_y','w','h','obj_id','confidence'])
#    res.to_csv('bbox_res/'+media_name+'.csv', index=False)
#    cnt+=1


# In[18]:


#cv2.destroyAllWindows()


# In[ ]:


for media_name in val_videos[30:]:
    if os.path.exists('/content/109-1-Data-Science-Project/bbox_res/'+media_name+'.csv'):
        start = time.time()
        res_list = []
        run(val_path, media_name)
        res = pd.DataFrame(res_list, columns=['video_id','timestamp','x1','y1','x2','y2','action_id','person_id'])
        res.to_csv('/content/drive/MyDrive/DataScienceProject/act_res/'+media_name+'.csv', index=False)
        print("Done in ", time.time()-start, 'seconds')


# In[ ]:





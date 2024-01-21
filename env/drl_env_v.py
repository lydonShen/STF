import os
import cv2
import numpy as np
import gym
from gym import spaces
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from env.predictor import COCODemo

class ViewSearchEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self, args, is_train=True):
        self.args = args
        self.cfg = self._get_cfg()

        if is_train:
            self.MAX_COUNT = 5
        else:
            self.MAX_COUNT = 10
        self.MAX_COUNT_THRESHOLD = 3
        self.NUM_CLASS = 12
        self.NUM_ACTION = 5
        self.COMPRESSION_RATIO = 4

        self.thresholds_for_classes = [0.5] * self.NUM_CLASS
        self.detector = self._get_detector()
        self.dataset, self.dataset_length = self._get_image_loader(is_train)
        self.position_dict = {1 : {1 : 0, 2 : 1, 3 : 2, 4 : 3}, 
                              2 : {2 : 4, 3 : 5, 4 : 6}, 
                              3 : {2 : 7, 3 : 8, 4 : 9}}

        self.image_id = 0
        self.current_image_name = None
        self.current_longitude = None
        self.current_latitude = None
        self.history_action = None
        self.state = None
        self.benefit = None
        self.mu = 0.5
        self.sigma2 = 0.01

        self.action_space = spaces.Discrete(self.NUM_ACTION)
        self.feature_length1 = (100, 180, np.int(256 / self.COMPRESSION_RATIO))
        self.feature_length2 = (50 , 90 , np.int(256 / self.COMPRESSION_RATIO))
        self.feature_length3 = (25 , 45 , np.int(256 / self.COMPRESSION_RATIO))
        self.feature_length4 = (13 , 23 , np.int(256 / self.COMPRESSION_RATIO))
        self.feature_length5 = (7  , 12 , np.int(256 / self.COMPRESSION_RATIO))
        self.feature_length6 = (10, )
        self.feature_length7 = (self.MAX_COUNT_THRESHOLD * self.NUM_ACTION, )

    def _get_cfg(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        cfg.MODEL.WEIGHT = self.args.weights
        cfg.freeze()
        return cfg

    def _get_detector(self):
        return COCODemo(self.cfg, confidence_thresholds_for_classes=self.thresholds_for_classes,
                        min_image_size=self.args.min_image_size)

    def _get_image_loader(self, is_train=True):
        data_loader = make_data_loader(self.cfg, is_train=is_train, is_distributed=False)

        if is_train:
            dataset = data_loader.dataset
            dataser_length = len(dataset)
        else:
            dataset = data_loader[0].dataset
            dataser_length = len(dataset)

        return dataset, dataser_length

    def _get_non_action_vector(self):
        c_lon = self.current_longitude
        c_lat = self.current_latitude

        non_action = np.zeros(self.NUM_ACTION)
        if c_lon == 1:
            non_action[1] = -np.inf
        elif c_lon == 3:
            non_action[2] = -np.inf

        if c_lat == 1:
            non_action[3] = -np.inf
        elif c_lat == 4:
            non_action[4] = -np.inf

        if c_lon == 1 and c_lat == 1:
            non_action[2] = -np.inf

        return non_action

    def update_threshold(self, benefit):
        self.mu = np.clip(self.mu * 0.9999 + benefit * 0.0001, 0, 1)
        self.sigma2 = self.sigma2 * 0.9999 + (benefit-self.mu)**2 * 0.0001

    def get_init_position(self):
        self.current_image_name = self.dataset.get_img_info(self.image_id)['file_name'].split('.')[0]
        self.current_longitude = np.int(self.current_image_name[3])
        self.current_latitude = np.int(self.current_image_name[4])
        return None

    def get_history_action_feature(self):
        history_action_features = np.ones(self.MAX_COUNT_THRESHOLD * self.NUM_ACTION)
        if len(self.history_action) != 0:
            for idx, act in enumerate(self.history_action):
                history_action_features[idx * self.NUM_ACTION + act] = 0
        return history_action_features

    def get_aggregated_feature(self, feature):
        feature = torch.reshape(feature, (1, self.COMPRESSION_RATIO, np.int(256 / self.COMPRESSION_RATIO),) + feature.shape[-2:])
        feature = torch.mean(feature, 1)[0]
        return feature
        # return torch.cat([torch.max(feature, 1)[0].unsqueeze(0), torch.mean(feature, 1, keepdim=True), torch.min(feature, 1)[0].unsqueeze(0)], 1)

    def get_position_feature(self):
        f1 = np.eye(10)[self.position_dict[self.current_longitude][self.current_latitude]]
        return f1

    def calculate_state(self, features):
        state = []
        for f in features:
            af = self.get_aggregated_feature(f)
            state.append(np.transpose(af.cpu().numpy().squeeze(), (1, 2, 0)))
        pf = self.get_position_feature()
        haf = self.get_history_action_feature()
        state.append(pf)
        state.append(haf)
        return state

    def calculate_trigger_state(self):
        state = []
        for s in self.state[:-1]:
            # state.append(np.zeros(s.shape))
            state.append(s)
        haf = self.get_history_action_feature()
        state.append(haf)
        return state

    def set_position_by_action(self, action):
        c_lon = self.current_longitude
        c_lat = self.current_latitude
        termination = False

        if action == 1:
            c_lon = c_lon - 1
        elif action == 2:
            c_lon = c_lon + 1
        elif action == 3:
            c_lat = c_lat - 1
            if c_lat == 1:
                c_lon = 1
        elif action == 4:
            c_lat = c_lat + 1
            if c_lat == 2 and c_lon == 1:
                c_lon = 2
        else:
            termination = True

        self.current_image_name = self.current_image_name[:3] + str(c_lon) + str(c_lat) + self.current_image_name[-2:]
        self.current_longitude = c_lon
        self.current_latitude = c_lat

        return termination
    
    def iou(self, rec1, rec2):
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        sum_area = S_rec1 + S_rec2

        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0

    def calculate_benefit(self, pred, gt):

        pr_bboxes = pred.bbox.cpu().numpy()
        gt_bboxes = gt.bbox.cpu().numpy()
        pr_labels = pred.get_field('labels').cpu().numpy()
        gt_labels = gt.get_field('labels').cpu().numpy()
        pr_scores = pred.get_field('scores').cpu().numpy()
        hit = np.zeros(pr_scores.shape)

        for idx, pr_bbox in enumerate(pr_bboxes):
            gt_bboxes_with_same_class = gt_bboxes[pr_labels[idx]==gt_labels]
            for gt_bbox in gt_bboxes_with_same_class:
                if self.iou(pr_bbox, gt_bbox) >= 0.5:
                    hit[idx] = 1
                    continue

        return 2 * np.sum(pr_scores*hit) / (pr_bboxes.shape[0] + gt_bboxes.shape[0])

    def calculate_reward(self, t2_benefit):

        t1_benefit = self.benefit
        reward = np.sign(t2_benefit - t1_benefit)

        if reward == 0:
            reward = -1.

        return reward

    def calculate_trigger_reward(self):

        benefit = self.benefit
        if benefit >= self.mu:
            return 2.
        elif benefit >= self.mu + 0.5 * np.sqrt(self.sigma2):
            return 3.
        elif benefit >= self.mu + np.sqrt(self.sigma2):
            return 4.
        elif benefit >= self.mu + 1.5 * np.sqrt(self.sigma2):
            return 5.
        else:
            return -3.

    def reset(self):

        self.counts = 0
        self.history_action = []
        init_img = np.array(self.dataset[self.image_id][0])
        init_gt = self.dataset[self.image_id][1]
        self.get_init_position()

        features, locations, box_cls, box_regression, centerness, image_sizes = self.detector.compute_state(init_img)
        init_bbox = self.detector.compute_bbox_from_state(init_img, features, locations, box_cls, box_regression, centerness, image_sizes)

        self.state = self.calculate_state(features)
        self.benefit = self.calculate_benefit(init_bbox, init_gt)
        self.update_threshold(self.benefit)
        self.image_id = (self.image_id + 1) % self.dataset_length

        # print(self.current_image_name)
        # self.bbox_visualize(init_img, init_bbox, 'init.jpg')

        return self.state

    def step(self, action):

        done = self.set_position_by_action(action)

        self.history_action.insert(0, action)
        if len(self.history_action) > self.MAX_COUNT_THRESHOLD:
            self.history_action.pop()

        if not done:

            current_image_id = self.dataset.get_index_from_id(np.int(self.current_image_name))
            current_img = np.array(self.dataset[current_image_id][0])
            current_gt = self.dataset[current_image_id][1]

            features, locations, box_cls, box_regression, centerness, image_sizes = self.detector.compute_state(current_img)
            current_bbox = self.detector.compute_bbox_from_state(current_img, features, locations, box_cls, box_regression, centerness, image_sizes)

            # print(self.current_image_name)
            # self.bbox_visualize(current_img, current_bbox, 'current.jpg')

            self.state = self.calculate_state(features)
            current_benefit = self.calculate_benefit(current_bbox, current_gt)
            reward = self.calculate_reward(current_benefit)
            self.benefit = current_benefit
            self.update_threshold(self.benefit)

        else:
            self.state = [i.squeeze() for i in self.state]
            self.state = self.calculate_trigger_state()
            reward = self.calculate_trigger_reward()

        self.counts += 1
        done = done or (self.counts >= self.MAX_COUNT)

        return self.state, reward, done, {}
        
    def render(self, demo_im_names="demo/images/", mode='human'):
        for im_name in os.listdir(demo_im_names):
            img = cv2.imread(os.path.join(self.dataset.root, im_name))
            if img is None:
                continue
            composite = self.detector.run_on_opencv_image(img)
            cv2.imshow(im_name, composite)
        print("Press any keys to exit ...")
        cv2.waitKey()
        self.close()
        return None
        
    def close(self):
        cv2.destroyAllWindows()
        return None

    def heatmap_visualize(self, heatmap, save_dir):
        heatmap = torch.sigmoid(heatmap)
        cv2.imwrite(save_dir, np.int32(heatmap.cpu().numpy()*256))
        return None

    def bbox_visualize(self, image, bbox, save_dir):
        composite = self.detector.visualize_bbox(image, bbox)
        cv2.imwrite(save_dir, composite)
        return None
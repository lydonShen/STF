import os
import cv2
import numpy as np
import gym
from gym import spaces
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from env.predictor import COCODemo
from teacher_net import SiameseTeacherModel

class ViewScaleBrightnessSearchEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self, args, is_train=False):
        self.args = args
        self.cfg = self._get_cfg()
        if is_train:
            self.MAX_COUNT = 5
        else:
            self.MAX_COUNT = 10
        self.MAX_COUNT_THRESHOLD = 3
        self.NUM_CLASS = 5
        self.NUM_ACTION = 9
        self.COMPRESSION_RATIO = 4
        self.thresholds_for_classes = [0.5] * self.NUM_CLASS
        self.detector = self._get_detector()
        self.dataset, self.dataset_length = self._get_image_loader(is_train)
        self.position_dict = {1 : {1 : 0, 2 : 1, 3 : 2, 4 : 3}, 
                              2 : {2 : 4, 3 : 5, 4 : 6}, 
                              3 : {2 : 7, 3 : 8, 4 : 9}}
        self.image_id = 0
        self.feature = None
        self.init_bbox = None
        self.current_image_name = None
        self.current_longitude = None
        self.current_latitude = None
        self.current_scale = None
        self.current_brightness = None
        self.history_action = None
        self.state = None
        self.benefit = None
        self.mu = 0.5
        self.sigma2 = 0.01
        self.ac_count = None
        self.action_space = spaces.Discrete(self.NUM_ACTION)
        self.teacher = self._get_SiameseTeacherModel('/models/teachernet_model.pth')

    def _get_cfg(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.merge_from_list(self.args.opts)
        cfg.MODEL.WEIGHT = self.args.weights
        cfg.freeze()
        return cfg

    def _get_detector(self):
        return COCODemo(self.cfg, confidence_thresholds_for_classes=self.thresholds_for_classes,
                        min_image_size=self.args.min_image_size)

    def _get_SiameseTeacherModel(self,dir=None):
        model=SiameseTeacherModel()
        model.load_state_dict(torch.load(dir))
        model.eval() 
        return model

    def _get_image_loader(self, is_train=False):
        data_loader = make_data_loader(self.cfg, is_train=is_train, is_distributed=False)
        if is_train:
            dataset = data_loader.dataset
            dataser_length = len(dataset)
        else:
            dataset = data_loader[0].dataset
            dataser_length = len(dataset)
        print(dataser_length)
        return dataset, dataser_length

    def _get_non_action_vector(self):
        c_lon = self.current_longitude
        c_lat = self.current_latitude
        c_sca = self.current_scale
        c_bri = self.current_brightness

        non_action = np.zeros(self.NUM_ACTION)
        if c_lon == 1:
            non_action[1] = -np.inf
        elif c_lon == 3:
            non_action[2] = -np.inf

        if c_lat == 1:
            non_action[3] = -np.inf
        elif c_lat == 4:
            non_action[4] = -np.inf

        if c_sca == 1:
            non_action[5] = -np.inf
        elif c_sca == 5:
            non_action[6] = -np.inf

        if c_bri == 1:
            non_action[7] = -np.inf
        elif c_bri == 5:
            non_action[8] = -np.inf

        if c_lon == 1 and c_lat == 1:
            non_action[2] = -np.inf
        return non_action

    def get_init_position(self,current_name):
        self.current_image_name = current_name
        self.current_longitude = np.int(self.current_image_name[3])
        self.current_latitude = np.int(self.current_image_name[4])
        self.current_scale = np.int(self.current_image_name[5])
        self.current_brightness = np.int(self.current_image_name[6])
        return None

    def set_position_by_action_greedy(self, action):
        c_lon = self.current_longitude
        c_lat = self.current_latitude
        c_sca = self.current_scale
        c_bri = self.current_brightness
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
        elif action == 5:
            c_sca = c_sca - 1
        elif action == 6:
            c_sca = c_sca + 1
        elif action == 7:
            c_bri = c_bri - 1
        elif action == 8:
            c_bri = c_bri + 1
        elif action == 9:
            c_bri = c_bri + 0
        else:
            return self.current_image_name
        current_image_name = self.current_image_name[:3] + str(c_lon) + str(c_lat) + str(c_sca) + str(c_bri)
        return current_image_name

    def step_ran(self,action):
        done = self.set_position_by_action(action)
        if not done:
            while not os.path.exists(self.testdir + self.current_image_name + ".npy"):
                done =self.set_position_by_de_action(action)
                action = np.random.randint(1, 9)
                done = self.set_position_by_action(action)
                if done:
                    reward = self.calculate_trigger_reward()
                    self.counts += 1
                    done = done or (self.counts >= self.MAX_COUNT)
                    return self.state, reward, done

        if not done:
            state = np.load(self.testdir + self.current_image_name + ".npy")
            self.state = torch.from_numpy(state)
            current_benefit = self.benefitdict[self.current_image_name]
            benefit = float(current_benefit)
            reward = self.calculate_reward(benefit)
            self.benefit = benefit
            self.update_threshold(self.benefit)
        else:
            reward = self.calculate_trigger_reward()

        self.counts += 1
        done = done or (self.counts >= self.MAX_COUNT)
        return self.state, reward, done

    def step_greedy(self):
        non_action = self._get_non_action_vector()
        available_action = np.argmax(np.eye(non_action.shape[0])[non_action > -1], 1)[1:]
        max_benefit = self.benefit
        max_benefit_action = 0
        for aa in available_action:
            current_image_name = self.set_position_by_action_greedy(aa)
            current_benefit = float(self.benefitdict[str(current_image_name)])
            self.update_threshold(current_benefit)
            if max_benefit < current_benefit:
                max_benefit_action = aa
                max_benefit = current_benefit
        return max_benefit_action

    def set_position_by_de_action(self, action):
        c_lon = self.current_longitude
        c_lat = self.current_latitude
        c_sca = self.current_scale
        c_bri = self.current_brightness
        termination = False
        if action == 1:
            c_lon = c_lon + 1
        elif action == 2:
            c_lon = c_lon - 1
        elif action == 3:
            c_lat = c_lat + 1
            if c_lat == 2 and c_lon == 1:
                c_lon = 2
        elif action == 4:
            c_lat = c_lat - 1
            if c_lat == 1:
                c_lon = 1
        elif action == 5:
            c_sca = c_sca + 1
        elif action == 6:
            c_sca = c_sca - 1
        elif action == 7:
            c_bri = c_bri + 1
        elif action == 8:
            c_bri = c_bri - 1
        elif action == 9:
            c_bri = c_bri + 0
        else:
            termination = True
        self.current_image_name = self.current_image_name[:3] + str(c_lon) + str(c_lat) + str(c_sca) + str(c_bri)
        self.current_longitude = c_lon
        self.current_latitude = c_lat
        self.current_scale = c_sca
        self.current_brightness = c_bri
        return termination

    def get_history_action_feature(self):
        history_action_features = np.ones(self.MAX_COUNT_THRESHOLD * self.NUM_ACTION)
        if len(self.history_action) != 0:
            for idx, act in enumerate(self.history_action):
                history_action_features[idx * self.NUM_ACTION + act] = 0
        return history_action_features

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
        c_sca = self.current_scale
        c_bri = self.current_brightness
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
        elif action == 5:
            c_sca = c_sca - 1
        elif action == 6:
            c_sca = c_sca + 1
        elif action == 7:
            c_bri = c_bri - 1
        elif action == 8:
            c_bri = c_bri + 1
        elif action == 9:
            c_bri = c_bri + 0
        else:
            termination = True

        self.current_image_name = self.current_image_name[:3] + str(c_lon) + str(c_lat) + str(c_sca) + str(c_bri)
        self.current_longitude = c_lon
        self.current_latitude = c_lat
        self.current_scale = c_sca
        self.current_brightness = c_bri
        return termination

    def update_threshold(self, benefit):
        self.mu = np.clip(self.mu * 0.9999 + benefit * 0.0001, 0, 1)
        self.sigma2 = self.sigma2 * 0.9999 + (benefit - self.mu) ** 2 * 0.0001

    def calculate_reward_loss(self, t2_feature, t2_init_bbox):
        t1_benefit = self.feature
        reward = self.teacher(t1_benefit, t2_feature,t2_init_bbox)
        return reward

    def calculate_trigger_reward_loss(self,t2_feature, t2_state):
        t1_benefit = self.feature
        reward = self.teacher(t1_benefit, t2_feature,t2_state)
        return reward

    def calculate_reward(self, t2_benefit):
        t1_benefit = self.t_feature
        reward = self.teacher(t1_benefit,t2_benefit)
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
        self.get_init_position()

        features, locations, box_cls, box_regression, centerness, image_sizes = self.detector.compute_state(init_img)
        init_bbox = self.detector.compute_bbox_from_state(init_img, features, locations, box_cls, box_regression,
                                                          centerness, image_sizes)

        self.state = self.calculate_state(features)
        self.feature = features
        self.init_bbox = init_bbox
        self.image_id = (self.image_id + 1) % self.dataset_length
        return self.state

    def step(self, action):
        done = self.set_position_by_action(action)
        self.history_action.insert(0, action)
        if len(self.history_action) > self.MAX_COUNT_THRESHOLD:
            self.history_action.pop()
        if not done:
            current_image_id = self.dataset.get_index_from_id(np.int(self.current_image_name))
            current_img = np.array(self.dataset[current_image_id][0])

            features, locations, box_cls, box_regression, centerness, image_sizes = self.detector.compute_state(
                current_img)
            current_bbox = self.detector.compute_bbox_from_state(current_img, features, locations, box_cls,
                                                                 box_regression, centerness, image_sizes)

            self.state = self.calculate_state(features)
            self.feature = features
            self.init_bbox = current_bbox
            reward = self.calculate_reward_loss(self.feature,self.init_bbox)

        else:
            self.state = [i.squeeze() for i in self.state]
            self.state = self.calculate_trigger_state()
            reward = self.calculate_trigger_reward_loss(self.feature,self.state)
        self.counts += 1
        done = done or (self.counts >= self.MAX_COUNT)
        statename =self.current_image_name

        return self.state, reward, done ,statename
        
    def set_position_by_action_greedy(self, action):
        c_lon = self.current_longitude
        c_lat = self.current_latitude
        c_sca = self.current_scale
        c_bri = self.current_brightness

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
        elif action == 5:
            c_sca = c_sca - 1
        elif action == 6:
            c_sca = c_sca + 1
        elif action == 7:
            c_bri = c_bri - 1
        elif action == 8:
            c_bri = c_bri + 1
        else:
            return self.current_image_name

        current_image_name = self.current_image_name[:3] + str(c_lon) + str(c_lat) + str(c_sca) + str(c_bri)
        return current_image_name

    def render(self, demo_im_names="demo/images/", mode='human'):
        for im_name in os.listdir(demo_im_names):
            img = cv2.imread(os.path.join(self.dataset.root, im_name))
            if img is None:
                continue
            composite = self.detector.run_on_opencv_image(img)
            cv2.imshow(im_name, composite)
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


    
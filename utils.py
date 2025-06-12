import itertools
import logging
import numpy as np
import tensorflow.compat.v1 as tf
import time
import os
import pandas as pd
import subprocess
from tensorflow.keras.utils import Progbar 
import tqdm
def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []
        self.output_path = output_path
        self.env.train_mode = True
        self.ckpt_interval = self.global_counter.total_step // 10
        self.model_dir = self.output_path + 'model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def _get_policy(self, ob, done, mode='train'):
        # 1) MA2C 系列
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
            action = []
            for pi in policy:
                if mode == 'train':
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action.append(np.argmax(pi))
            return policy, np.array(action)

        # 2) MAPPO（Multi‐Agent PPO）
        elif self.agent == 'mappo':
            # 将 ob 传给 MAPPOPolicy，得到 N_agent 个 pi
            # MAPPOPolicy.forward(sess, obs_in, done_in, agent_i=i, out_type='p')
            obs_in = np.expand_dims(np.array(ob, dtype=np.float32), axis=1)  # (N_agent, 1, n_s)
            done_in = np.array([done], dtype=np.float32)                     # (1,)
            all_pi = []
            all_action = []
            for i in range(self.env.n_agent):
                pi_i = self.model.forward(obs_in, done_in, agent_i=i, out_type='p')  # 形状 (1, n_a)
                pi_i = pi_i[0]  # 取成 (n_a,)
                all_pi.append(pi_i)
                if mode == 'train':
                    a_i = np.random.choice(len(pi_i), p=pi_i)
                else:
                    a_i = np.argmax(pi_i)
                all_action.append(a_i)
            return all_pi, np.array(all_action, dtype=np.int32)

        # 3) 其余算法（例如 IA2C / Greedy / etc.）
        else:
            policy = self.model.forward(ob, done)
            action = []
            for pi in policy:
                if mode == 'train':
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action.append(np.argmax(pi))
            return policy, np.array(action)

    def _get_value(self, ob, done, action):
        # 1) MA2C 系列
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
            return value

        # 2) MAPPO
        elif self.agent == 'mappo':
            # 直接让 MAPPOPolicy 分别对每个 agent 计算 V(s)
            obs_in = np.expand_dims(np.array(ob, dtype=np.float32), axis=1)  # (N_agent, 1, n_s)
            done_in = np.array([done], dtype=np.float32)
            all_v = []
            for i in range(self.env.n_agent):
                v_i = self.model.forward(obs_in, done_in, agent_i=i, out_type='v')  # 返回 shape (1,)
                all_v.append(v_i[0])
            return np.array(all_v, dtype=np.float32)

        # 3) 其它算法（IA2C / Greedy 等）
        else:
            self.naction = self.env.get_neighbor_action(action)
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
            return value


    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        self.data.append(log)
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            # 1. 获取本次策略和动作
            if self.agent == 'mappo':
                policy, action = self._get_policy(ob, done, mode='train')  # policy:list of π_i, action: array(N_agent,)
            elif self.agent.startswith('ma2c'):
                self.ps = self.env.get_fingerprint()
                policy, action = self._get_policy(ob, done)
            else:
                policy, action = self._get_policy(ob, done)

            # 2. 计算 V(s)
            if self.agent == 'mappo':
                value = self._get_value(ob, done, action)  # 返回 shape (N_agent,)
            else:
                value = self._get_value(ob, done, action)

            # 3. 与环境交互
            self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            self.episode_rewards.append(global_reward)
            global_step = self.global_counter.next()
            if global_step % self.ckpt_interval == 0:
                self.model.save(self.model_dir, global_step)
                logging.info(f"[Checkpoint] saved at step {global_step:,}")
            self.cur_step += 1

            # 4. 存 buffer
            if self.agent == 'mappo':
                # 对 MAPPO 来说，把 action 当作 naction feed 给 centralized critic
                # add_transition(ob, naction, action, reward, value, done)
                self.model.add_transition(ob, action, action, reward, value, done)
            elif self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)

            # 5. logging & 终止检查
            if self.global_counter.should_log():
                logging.info(
                    f"Training: global step {global_step}, episode step {self.cur_step}, "
                    f"ob: {ob}, a: {action}, pi: {policy}, r: {global_reward:.2f}, "
                    f"train r: {np.mean(reward):.2f}, done: {done}"
                )
            if done:
                break
            ob = next_ob

        # 6. 计算本轮最后一个 state 的 R (bootstrap)
        if done:
            R = np.zeros(self.model.n_agent, dtype=np.float32)
        else:
            if self.agent == 'mappo':
                _, action = self._get_policy(ob, done, mode='train')
                R = self._get_value(ob, done, action)  # shape (N_agent,)
            else:
                _, action = self._get_policy(ob, done)
                R = self._get_value(ob, done, action)
        return ob, done, R


    def perform(self, test_ind, gui=False):
        ob = self.env.reset(gui=gui, test_ind=test_ind)
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                # in on-policy learning, test policy has to be stochastic
                if self.env.name.startswith('atsc'):
                    policy, action = self._get_policy(ob, done)
                else:
                    # for mission-critic tasks like CACC, we need deterministic policy
                    policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        while not self.global_counter.should_stop():
            # np.random.seed(self.env.seed)
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            self.episode_rewards = []
            while True:
                ob, done, R = self.explore(ob, done)
                dt = self.env.T - self.cur_step
                global_step = self.global_counter.cur_step
                self.model.backward(R, dt, self.summary_writer, global_step)
                # termination
                if done:
                    self.env.terminate()
                    break
            rewards = np.array(self.episode_rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # NOTE: for CACC we have to run another testing episode after each
            # training episode since the reward and policy settings are different!
            if not self.env.name.startswith('atsc'):
                self.env.train_mode = False
                mean_reward, std_reward = self.perform(-1)
                self.env.train_mode = True
            self._log_episode(global_step, mean_reward, std_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path, gui=False):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.gui = gui

    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()

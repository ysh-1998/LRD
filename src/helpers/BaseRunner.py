# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn
import pickle

from utils import utils
from models.BaseModel import BaseModel


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--early_stop', type=int, default=5,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=5,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='[5,10,50]',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='["NDCG","HR"]',
                            help='metrics: NDCG, HR')
        return parser

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list, save_rank=0) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K values list
        :param metrics: metrics string list
        :return: a result dict, the keys are metrics@topk
        """
        evaluations = dict()
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        if save_rank:
            with open('gt_rank.pkl','wb') as f:
                pickle.dump(gt_rank,f)
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        return evaluations

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = eval(args.topk)
        self.metrics = [m.strip().upper() for m in eval(args.metric)]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric
        self.only_predict = args.only_predict

        self.time = None  # will store [start_time, last_step_time]

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adadelta':
            logging.info("Optimizer: Adadelta")
            optimizer = torch.optim.Adadelta(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(
                model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    def train(self, model: nn.Module, data_dict: Dict[str, BaseModel.Dataset]) -> NoReturn:
        main_metric_results, dev_results, test_results = list(), list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                loss = self.fit(model, data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev and test results
                dev_result = self.evaluate(model, data_dict['dev'], self.topk[:1], self.metrics)
                test_result = self.evaluate(model, data_dict['test'], self.topk[:1], self.metrics)
                testing_time = self._check_time()
                dev_results.append(dev_result)
                test_results.append(test_result)
                main_metric_results.append(dev_result[self.main_metric])

                logging.info("Epoch {:<5} loss={:<.4f} [{:<.1f} s]\t dev=({}) test=({}) [{:<.1f} s] ".format(
                             epoch + 1, loss, training_time, utils.format_metric(dev_result),
                             utils.format_metric(test_result), testing_time))

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 1):
                    model.save_model()
                if self.early_stop and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) test=({}) [{:<.1f} s] ".format(
                     best_epoch + 1, utils.format_metric(dev_results[best_epoch]),
                     utils.format_metric(test_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()

    def fit(self, model: nn.Module, data: BaseModel.Dataset, epoch=-1) -> float:
        gc.collect()
        torch.cuda.empty_cache()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        data.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        all_losses_list = list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            loss, all_losses = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
            all_losses_list.append(all_losses)
        all_losses = torch.tensor(all_losses_list).mean(dim=0).detach().cpu().data.numpy()
        logging.info(f"all_losses: {str(all_losses)}")
        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > 20 and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > 20:
            return True
        return False

    def evaluate(self, model: nn.Module, data: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions = self.predict(model, data)
        if self.only_predict:
            return self.evaluate_method(predictions, topks, metrics,save_rank=1)
        else:
            return self.evaluate_method(predictions, topks, metrics)

    def predict(self, model: nn.Module, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions order: [[1,3,4], [2,5,6]]
        """
        model.eval()
        predictions = list()
        lrd_dict = {"history_target": [], "predict": []}
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            out_dict = model(utils.batch_to_gpu(batch, model.device))
            prediction = out_dict['prediction']
            predictions.extend(prediction.cpu().data.numpy())
            if self.only_predict:
                lrd_dict['history_target'].extend(out_dict["his_i_ids"].cpu().data.numpy())
                lrd_dict['predict'].extend(out_dict["target_attention"].cpu().data.numpy())
        if self.only_predict:
            lrd_dict['history_target'] = np.array(lrd_dict['history_target'])
            lrd_dict['predict'] = np.array(lrd_dict['predict'])
            with open('lrd_result.pkl','wb') as f:
                pickle.dump(lrd_dict,f)
        return np.array(predictions)

    def print_res(self, model: nn.Module, data: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(model, data, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str

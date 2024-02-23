from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger

import os
import wandb
import numpy as np

class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg

        self.tboard_log_dir = os.path.join(cfg.log_dir, 'tboard_logs')
        os.makedirs(self.tboard_log_dir,exist_ok=True)

    def _log_metric(self, tag, scalar_value, global_step, tboard_writer, cfg=None):
        if cfg is None or cfg.log_in_wandb:
            try:
                wandb.log({tag: scalar_value}, step=global_step)
            except Exception as e:
                print(f"WandB log failed for tag '{tag}': {e}")
        tboard_writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.cfg.sum_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {}] ".format(self.total_steps+1, self.scheduler.get_last_lr())
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        loguru_logger.info(training_str + metrics_str)

        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)

        for k in self.running_loss:
            # self.writer.add_scalar(k, self.running_loss[k]/self.cfg.sum_freq, self.total_steps)
            self._log_metric(tag=k, scalar_value=self.running_loss[k]/self.cfg.sum_freq, global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)

            self.running_loss[k] = 0.0

    def push(self, metrics, model, current_lr, train_loss, curr_epoch):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}

            var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
            var_cnt = len(var_sum)
            var_sum = np.sum(var_sum)
            var_avg = var_sum.item()/var_cnt
            # by suraj
            var_norm = [var.norm().item() for var in model.parameters() if var.requires_grad]
            
            self._log_metric(tag="Training loss", scalar_value=train_loss, global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)
            self._log_metric(tag='Learning Rate', scalar_value=current_lr, global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)
            self._log_metric(tag='var sum average', scalar_value=var_avg, global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)
            self._log_metric(tag='var norm average', scalar_value=np.mean(var_norm), global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)
            self._log_metric(tag='Epoch', scalar_value=curr_epoch, global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)

    def write_dict(self, results, val_dataset, cfg):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            # self.writer.add_scalar(key, results[key], self.total_steps)
            self._log_metric(tag=f"Val_{val_dataset}_{key}", scalar_value=results[key], global_step=self.total_steps, tboard_writer=self.writer, cfg=self.cfg)


    def close(self):
        self.writer.close()


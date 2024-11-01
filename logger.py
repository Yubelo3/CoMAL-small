from torch.utils.tensorboard.writer import SummaryWriter
from time import gmtime, strftime
import os
import torch


class TBWriter:
    def __init__(self, task: str, log_prefix="logs/",ckpt_prefix="ckpt/") -> None:
        if not os.path.exists(log_prefix):
            os.mkdir(log_prefix)
        if not os.path.exists(ckpt_prefix):
            os.mkdir(ckpt_prefix)
        log_task_path = log_prefix+task
        ckpt_task_path = ckpt_prefix+task
        if not log_task_path.endswith("/"):
            log_task_path += "/"
        if not ckpt_task_path.endswith("/"):
            ckpt_task_path += "/"
        if not os.path.exists(log_task_path):
            os.mkdir(log_task_path)
        if not os.path.exists(ckpt_task_path):
            os.mkdir(ckpt_task_path)
        cur_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        log_path = log_task_path+cur_time+"/"
        ckpt_path= ckpt_task_path+cur_time+"/"
        os.mkdir(log_path)
        os.mkdir(ckpt_path)
        self.log_path=log_path
        self.ckpt_path=ckpt_path
        self.writer = SummaryWriter(log_path)
        print("logging at "+log_path)

    def add_scalar(self, tag, scalar, step):
        self.writer.add_scalar(tag, scalar, step)
    
    def save_ckpt(self, models:dict, step):
        for k,v in models.items():
            torch.save(v.state_dict(),f"{self.ckpt_path}{k}-{step}.pt")

from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(list(set(milestones)))  # 去重排序
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1  # 累计步骤数
        if self.step_count in self.milestones:  # 检查当前步是否为里程碑
            self.optimizer.init_lr *= self.gamma  # 按衰减系数更新学习率

class ExponentialLR(scheduler):
    pass
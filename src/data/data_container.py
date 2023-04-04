import copy
import random
import typing
import numpy as np
import torch
from src.apis.extensions import Functional

Best = ["SAPAsiaPhone","SAPAfricaPhone","SAPAmericaWatch","SAPEuropeLock","IBMAsiaLock","IBMAfricaPhone","IBMAmericaSecurity","IBMEuropePhone","CiscoAsiaPhone","CiscoAfricaWatch","CiscoAmericaSecurity","CiscoEuropeWatch","PTCAsiaWatch","PTCAfricaSecurity","PTCAmericaSecurity","PTCEuropeLock"]
class DataContainer(Functional):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def batch(self, batch_size):
        if len(self.x) == 0:
            return list()
        batch_data = list()
        batch_size = len(self.x) if batch_size <= 0 or len(self.x) < batch_size else batch_size
        for i in range(0, len(self.x), batch_size):
            batched_x = self.x[i:i + batch_size]
            batched_y = self.y[i:i + batch_size]
            batch_data.append((batched_x, batched_y))
        return batch_data

    def labels(self):
        return list(np.unique(self.y))

    def get(self):
        return self.x, self.y

    def is_empty(self):
        return self.x is None or len(self) == 0

    def __len__(self):
        return len(self.x)

    def is_tensor(self):
        return torch.is_tensor(self.x)

    def is_numpy(self):
        return type(self.x) == np.ndarray

    def as_tensor(self) -> 'DataContainer':
        if self.is_tensor():
            return self
        if self.is_numpy():
            return DataContainer(torch.from_numpy(self.x).float(), torch.from_numpy(self.y).long())
        return DataContainer(torch.tensor(self.x).float(), torch.tensor(self.y).long())

    def as_numpy(self, dtype=None) -> 'DataContainer':
        if self.is_tensor():
            return DataContainer(self.x.numpy(), self.y.numpy())
        if self.is_numpy():
            return self
        return DataContainer(np.asarray(self.x, dtype=dtype), np.asarray(self.y, dtype=dtype))

    def as_list(self) -> 'DataContainer':
        if self.is_numpy():
            return DataContainer(self.x.tolist(), self.y.tolist())
        if self.is_tensor():
            return DataContainer(self.x.numpy().tolist(), self.y.numpy().tolist())
        return self

    def split(self, train_freq, IoT=None) -> ('DataContainer', 'DataContainer'):
        total_size = len(self)
        train_size = int(total_size * train_freq)
        x_train = self.x[0:train_size]
        y_train = self.y[0:train_size]
        x_test = self.x[train_size:total_size]
        y_test = self.y[train_size:total_size]
        if IoT:
            y_test = self.pois(IoT=IoT, dtpois=y_test)
        return DataContainer(x_train, y_train), DataContainer(x_test, y_test)

    def poismain(self, datatype='kdd', IoT=None) -> 'DataContainer':
        if IoT:
            Region = IoT.Region
            dtype = IoT.Devt
            # print(Region, "\t", dtype)
            if Region == "Asia":
                if dtype == "Phone":
                    pois_rate = 0.3
                elif dtype == "Security":
                    pois_rate = 0.4
                elif dtype == "lock":
                    pois_rate = 0.5
                else:
                    pois_rate = 0
            elif Region == "Africa":
                if dtype == "watch":
                    pois_rate = 0.4
                elif dtype == "Security":
                    pois_rate = 0.5
                elif dtype == "lock":
                    pois_rate = 0.3
                else:
                    pois_rate = 0
            elif Region == "America":
                if dtype == "Phone":
                    pois_rate = 0.5
                elif dtype == "Security":
                    pois_rate = 0.4
                elif dtype == "watch":
                    pois_rate = 0.3
                else:
                    pois_rate = 0
            else:
                if dtype == "Phone":
                    pois_rate = 0.4
                elif dtype == "watch":
                    pois_rate = 0.3
                elif dtype == "lock":
                    pois_rate = 0.5
                else:
                    pois_rate = 0
        total_size = len(self.y)
        # print(total_size)
        pois_size = int(total_size * pois_rate)
        # print(pois_size)
        for i in range(0, pois_size):
            if datatype == 'mnist':
                r = random.randint(0, 9)
                while r == self.y[i]:
                    r = random.randint(0, 9)
                # print(self.y[i], "==>", r)
                self.y[i] = r
            else:
                r = random.randint(0, 4)
                while r == self.y[i]:
                    r = random.randint(0, 4)
                # print(self.y[i], "==>", r)
                self.y[i] = r
                # if self.y[i] == 0:
                #     self.y[i] = 1
                # else:
                #     self.y[i] = 0
        # x_train = self.x[0:pois_size]
        # y_train = self.y[0:pois_size]
        # x_test = self.x[pois_size:total_size]
        # y_test = self.y[pois_size:total_size]
        return DataContainer(self.x, self.y)

    def pois(self, pois_rate=None, datatype=None, IoT=None, dtpois=None) -> 'DataContainer':

        if IoT:
            Region = IoT.Region
            dtype = IoT.Devt
            producer = IoT.Producer
            # print((producer+Region+dtype))
            if (producer+Region+dtype) in Best:
                pois_rate = 0
                # print("pass the good!!!")
            else:
                pois_rate = 0.5
            pois_size = int(pois_rate*len(dtpois))
            for i in range(pois_size):
                r = random.randint(0, 9)
                while r == dtpois[i]:
                    r = random.randint(0, 9)
                # print(dtpois[i], "==>", r)
                dtpois[i] = r
            return dtpois
        else:
            total_size = len(self.y)
            # print(total_size)
            pois_size = int(total_size * pois_rate)
            # print(pois_size)
            for i in range(0, pois_size):
                if datatype == 'Mnist':
                    r = random.randint(0, 9)
                    while r == self.y[i]:
                        r = random.randint(0, 9)
                    self.y[i] = r
                else:
                    r = random.randint(0, 4)
                    while r == self.y[i]:
                        r = random.randint(0, 4)
                    self.y[i] = r
            # x_train = self.x[0:pois_size]
            # y_train = self.y[0:pois_size]
            # x_test = self.x[pois_size:total_size]
            # y_test = self.y[pois_size:total_size]
            return DataContainer(self.x, self.y)

    def shuffle(self, seed=None):
        dc = copy.deepcopy(self) if self.is_numpy() else self.as_numpy()
        permutation = np.random
        if seed is not None and isinstance(seed, int):
            permutation = permutation.RandomState(seed=seed)
        p = permutation.permutation(len(dc.x))
        return DataContainer(dc.x[p], dc.y[p])

    def filter(self, predictor: typing.Callable[[typing.List, float], bool]) -> 'DataContainer':
        current = self.as_list()
        new_x = []
        new_y = []
        for x, y in zip(current.x, current.y):
            if predictor(x, y):
                new_x.append(x)
                new_y.append(y)
        return self._from_list(new_x, new_y)

    def map(self, mapper: typing.Callable[[typing.List, int], typing.Tuple[typing.List, int]]) -> 'DataContainer':
        current = self.as_list()
        new_x = []
        new_y = []
        for x, y in zip(current.x, current.y):
            nx, ny = mapper(x, y)
            new_x.append(nx)
            new_y.append(ny)
        return self._from_list(new_x, new_y)

    def reshape(self, shape):
        return DataContainer(np.reshape(self.x, shape), self.y)

    def transpose(self, shape):
        return DataContainer(np.transpose(self.x, shape), self.y)

    def _from_list(self, x, y):
        new_dt = DataContainer(x, y)
        if self.is_numpy():
            return new_dt.as_numpy()
        if self.is_tensor():
            return new_dt.as_tensor()
        return new_dt

    def for_each(self, func: typing.Callable[[typing.List, float], typing.NoReturn]):
        for x, y in zip(self.x, self.y):
            func(x, y)

    def reduce(self, func: typing.Callable[[typing.Any, typing.List, float], typing.Any]) -> 'DataContainer':
        first = None
        for x, y in zip(self.x, self.y):
            first = func(first, x, y)
        return first

    def select(self, keys) -> 'DataContainer':
        current = self.as_list()
        new_x = []
        new_y = []
        for key in keys:
            new_x.append(current.x[key])
            new_y.append(current.y[key])
        return self._from_list(new_x, new_y)

    def concat(self, other) -> 'DataContainer':
        new_x = other.x if self.is_empty() else np.concatenate((self.x, other.x))
        new_y = other.y if self.is_empty() else np.concatenate((self.y, other.y))
        return DataContainer(new_x, new_y)

    def __repr__(self):
        return f'Size:{len(self)}, Unique:{np.unique(self.y)}, Features:{None if self.is_empty() else np.shape(self.x[0])}'

import logging
import random
import time
import xlsxwriter

import libs
import newgenerate
import Preference
import Match
from torch import nn
from libs.model.cv.resnet import resnet56
import libs.model.cv.cnn
from libs.model.linear.lr import LogisticRegression
from src.apis import lambdas
from src.apis.rw import IODict
from src.data.data_container import DataContainer
from src.data.data_distributor import LabelDistributor, ShardDistributor
from src.data.data_distributor import SizeDistributor
from src.data.data_loader import preload
from apps.experiments import federated_averaging as tfed
from apps.experiments import MyFunctions as MYF
from src.federated.components import trainers, aggregators, metrics, client_selectors
from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.events import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.subscribers import fed_plots
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.timer import Timer
from src.apis.extensions import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test')
logger.info('Start Test\'s....')

# for data poisining
def poison(dc: DataContainer, rate):
    total_size = len(dc)
    poison_size = int(total_size * rate)
    for i in range(0, poison_size):
        dc.y[i] = 0 if dc.y[i] != 0 else random.randint(1, 9)

# for splitting data
def splter(client_data, pois=True):
    poisc = 0
    train_data = Dict()
    test_data = Dict()
    for trainer_id, data in client_data.items():
        data = data.shuffle().as_tensor()
        train, test = data.split(0.7)
        train_data[trainer_id] = train
        if pois and poisc < 29:
            poison(test, 0.5)
        test_data[trainer_id] = test
        poisc += 1
    return train_data, test_data


# generate should take a dictionary of needed server with the number of clients needed by each
# plus the total number of iot devices
# save the data Partion for each data set
alldatapart = {}

# generate Iot and Fedservers
IotArr, FedArr, pardic = newgenerate.generate(2)

# print the FedServers
logger.info('Fed elements:')
for i in FedArr:
    logger.info('%s', i)

# prepare data dict for each IOT
dis = preload('cifar10', ShardDistributor(200, 3))

# fix the ids in data
new_dict = {}

for key in dis.keys():
    new_key = "IOT" + str(key)  # add letter "a" to the beginning of the key
    new_dict[new_key] = dis[key]

dis = new_dict
# split data and pois it
train_data, test_data = splter(dis)


def create_model(name):
    if name == 'resnet':
        return resnet56(10, 3, 32)
    else:
        global train_data
        global test_data
        # cifar10 data reduced to 1 dimension from 32,32,3. cnn32 model requires the image shape to be 3,32,32
        train_data = train_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
        test_data = test_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
        return libs.model.cv.cnn.Cifar10Model()


initialize_model = create_model('cnn')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('warmuppppp')

trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=5, optimizer='sgd',
                               criterion='cel', lr=0.1)

federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss(), device='cpu'),
    client_selector=client_selectors.All(),
    trainers_data_dict=train_data,
    test_data=test_data,
    initial_model=lambda: initialize_model,
    num_rounds=3,
    desired_accuracy=0.99,
    accepted_accuracy_margin=0.01,
    Selector=IotArr,
    DataName='cifar10'
)

federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
# federated.add_subscriber(EMDWeightDivergence(show_plot=False))
# federated.add_subscriber(SQLiteLogger('avg_2'))
# federated.add_subscriber(Resumable(IODict('./saved_models/test_avg'), save_ratio=1))
# federated.add_subscriber(fed_plots.RoundAccuracy(plot_ratio=0))
logger.info("----------------------")
logger.info("start federated Old")
logger.info("----------------------")
federated.start()
# end warmup to get std

logger.info('IoT elements:')
for i in IotArr:
    logger.info('%s', i)

# add to preference of Fed
random.shuffle(IotArr)
random.shuffle(FedArr)

Preference.FedPreference(IotArr, FedArr)
# choose the method 2 criterion (Sort) OR MCDM (calculateSw)
# Preference.calculateSwFed(FedArr)
Preference.sortFed(FedArr)
# add to preference of IOT
Preference.IoTPreference(IotArr, FedArr)
# choose the method 2 criterion (Sort) OR MCDM (calculateSw)
# Preference.calculateSwIot(IotArr)
Preference.sortIot(IotArr)
# pass the iot arr and fed to match as is then create function in each class iot and arr that return
# as result the name of the server with its preference same for iot...
# receive them put them in dictionaries run match then
# return as result 2 dictionaries and set the selected to preference if the FedServer
# call match
# for i in IotArr:
#     print(i.FSweightdic)
# quit()
IoTs_matching, Servers_m, Servers_matching, revenue = Match.federated_matching(IotArr, FedArr)
logger = logging.getLogger('Main')

# for i, j in Servers_matching.items():
#     logger.info('%s : %s', i, j)

workbook = xlsxwriter.Workbook("test1.xlsx")
TFedArr = []
tag = "icsf"

# start create federated
for i in FedArr:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')
    worksheet = workbook.add_worksheet(i.getName() + tag)
    worksheet.write('A1', i.getName() + ' ' + tag)
    worksheet.write('A2',
                    'MNIST DATA' + ' # of Clients ' + str(i.IOTNum) + ' # of Rounds ' + str(i.round))
    worksheet.write('A3', 'Round#')
    worksheet.write('B3', 'Accuracy')
    IOTLIST = i.preference
    trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=10, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    if isinstance(IOTLIST, int):
        IOTLIST = None
    else:
        pass

    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss(), device='cpu'),
        client_selector=client_selectors.All(),
        trainers_data_dict=train_data,
        test_data=test_data,
        num_rounds=3,
        initial_model=lambda: initialize_model,
        desired_accuracy=0.99,
        accepted_accuracy_margin=0.05,
        Selector=IOTLIST,
        FedServer=i,
        worksheet=worksheet
    )

    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    logger.info("----------------------")
    logger.info('%s', i.getName())
    logger.info("----------------------")
    federated.init()
    TFedArr.append(federated)
# federated created

for i in TFedArr:
    i.one_round()
    i.FedServer.round = i.FedServer.round - 1
# for i in IotArr:
#     i.FSweightdic = {}
#     i.preference = []
# logger.info('IoT elements:')
# for i in IotArr:
#     logger.info('%s', i)
MAX_Round = -1
for t in FedArr:
    if t.round > MAX_Round:
        MAX_Round = t.round

for count in range(MAX_Round):
    # add to preference of Fed
    random.shuffle(IotArr)
    random.shuffle(FedArr)

    Preference.FedPreference(IotArr, FedArr)
    # choose the method 2 criterion (Sort) OR MCDM (calculateSw)
    # Preference.calculateSwFed(FedArr)
    Preference.sortFed(FedArr)
    # add to preference of IOT
    Preference.IoTPreference(IotArr, FedArr)
    # choose the method 2 criterion (Sort) OR MCDM (calculateSw)
    # Preference.calculateSwIot(IotArr)
    Preference.sortIot(IotArr)
    # pass the iot arr and fed to match as is then create function in each class iot and arr that return
    # as result the name of the server with its preference same for iot...
    # receive them put them in dictionaries run match then
    # return as result 2 dictionaries and set the selected to preference if the FedServer
    # call match
    IoTs_matching, Servers_m, Servers_matching, revenue = Match.federated_matching(IotArr, FedArr)

    # logger.info('Matching Results:')
    # for i, j in Servers_matching.items():
    #     logger.info('%s : %s', i, j)

    for i in TFedArr:
        i.one_round()
        i.FedServer.round = i.FedServer.round - 1
    # for i in IotArr:
    #     i.FSweightdic = {}
    #     i.preference = []
    # if count == MAX_Round:
    #     time.sleep(0.2)
    # logger.info('IoT elements:')
    # for i in IotArr:
    #     logger.info('%s', i)

# quit()
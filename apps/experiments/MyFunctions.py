import logging
import sys
import xlsxwriter
from src.federated.subscribers import fed_plots

sys.path.append('../../')

from torch import nn
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.logger import FederatedLogger
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from src.federated.subscribers.timer import Timer
from src.data.data_distributor import LabelDistributor
from src.data.data_loader import preload
from libs.model.linear.lr import LogisticRegression
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager
from src.data import data_loader
from src.apis.extensions import Dict

def mnistFed(FedServer=None, client_data=None, tag=None, workbook=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    # logger.info('Generating Data --Started')
    # dist = LabelDistributor(100, 10, 600, 600)
    # print(f"{is_Dummy}")
    # quit()
    # dist = LabelDistributor(IOTLIST, 10, 600, 600, is_Dummy=is_Dummy)
    # params = ['client_number']
    # client_data = preload('mnist', dist)
    # client_data = preload(dataname, dist)
    # logger.info('Generating Data --Ended')
    # keep just the selected clients
    # test = preload('mnist10k', tag="mydata").as_tensor()
    # testing = {h: test for h in client_data.keys()}
    # testing = Dict(testing)
    # print(isinstance(testing, DataContainer))
    # print(testing)
    # quit()
    worksheet = workbook.add_worksheet(FedServer.getName() + tag)
    worksheet.write('A1', FedServer.getName() + ' ' + tag)
    worksheet.write('A2', 'MNIST DATA' + ' # of Clients ' + str(FedServer.IOTNum) + ' # of Rounds ' + str(FedServer.round))
    worksheet.write('A3', 'Round#')
    worksheet.write('B3', 'Accuracy')
    IOTLIST = FedServer.preference
    trainer_params = TrainerParams(trainer_class=trainers.CPUTrainer, batch_size=50, epochs=3, optimizer='sgd',
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
        client_selector=client_selectors.Random(1),
        trainers_data_dict=client_data,
        train_ratio=0.7,
        num_rounds=FedServer.round,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        desired_accuracy=0.99,
        accepted_accuracy_margin=0.01,
        Selector=IOTLIST,
        FedServer=FedServer,
        worksheet=worksheet
    )

    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    # federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    # federated.add_subscriber(EMDWeightDivergence(show_plot=False))
    # federated.add_subscriber(SQLiteLogger(FedServer.getName()+tag))
    # federated.add_subscriber(Resumable(IODict('./saved_models/test_avg'), save_ratio=1))
    federated.add_subscriber(fed_plots.RoundAccuracy(plot_ratio=0, plot_title=FedServer.getName()+' '+FedServer.getDName()+ ' '+tag))
    # federated.add_subscriber(fed_plots.RoundAccuracy(plot_ratio=0, plot_title=FedServer.getName()+' '+FedServer.getDName()+ ' '+tag+' C'+str(FedServer.IOTNum)+' R'+str(FedServer.round), save_dir='./'))
    logger.info("----------------------")
    logger.info('%s', FedServer.getName())
    logger.info("----------------------")
    federated.init()
    return federated


def KDDFED(FedServer=None, client_data=None, tag=None, workbook=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main'+tag)

    worksheet= workbook.add_worksheet(FedServer.getName()+tag)
    worksheet.write('A1', FedServer.getName() + ' ' + tag)
    worksheet.write('A2', 'KDD DATA' + ' # of Clients ' + str(FedServer.IOTNum) + ' # of Rounds ' + str(FedServer.round))
    worksheet.write('A3', 'Round#')
    worksheet.write('B3', 'Accuracy')
    # keep just the selected clients
    IOTLIST = FedServer.preference
    trainer_config = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=3, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_config,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(1),
        trainers_data_dict=client_data,
        train_ratio=0.7,
        num_rounds=FedServer.round,
        initial_model=lambda: LogisticRegression(41, 5),
        desired_accuracy=0.99,
        Selector=IOTLIST,
        FedServer=FedServer,
        worksheet=worksheet
    )

    federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    # federated.add_subscriber(SQLiteLogger('avg_22'))
    federated.add_subscriber(fed_plots.RoundAccuracy(plot_ratio=0, plot_title=FedServer.getName()+' '+FedServer.getDName()+ ' '+tag+str(FedServer.IOTNum)+' R'+str(FedServer.round), save_dir='./'))

    logger.info("----------------------")
    logger.info('%s', FedServer.getName())
    logger.info("----------------------")
    federated.init()
    return federated

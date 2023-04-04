import logging
import sys
from src.apis.extensions import Dict
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

def mnistdata(IOTLIST=None, is_Dummy=False, dataname='mnist', dataset=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    logger.info('Generating Data --Started')
    client_data = dataset
    # test = preload('mnist10k', tag="mydata").as_tensor()
    # testing = {h: test for h in client_data.keys()}
    # testing = Dict(testing)
    logger.info('Generating Data --Ended')

    trainer_params = TrainerParams(trainer_class=trainers.CPUTrainer, batch_size=50, epochs=3, optimizer='sgd',
                                   criterion='cel', lr=0.1)

    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss(), device='cpu'),
        client_selector=client_selectors.Random(1),
        trainers_data_dict=client_data,
        train_ratio=0.7,
        initial_model=lambda: LogisticRegression(28 * 28, 10),
        num_rounds=3,
        desired_accuracy=0.99,
        accepted_accuracy_margin=0.01,
        Selector=IOTLIST,
        DataName='mnist'
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

def kdddata(IOTLIST=None, is_Dummy=False, dataname='mnist', dataset=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    logger.info('Generating Data --Started')
    client_data = dataset
    # test = preload('kdd_test', tag="mykdata").as_tensor()
    # testing = {h: test for h in client_data.keys()}
    # testing = Dict(testing)
    logger.info('Generating Data --Ended')

    trainer_config = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=10, optimizer='sgd',
                                   criterion='cel', lr=0.1)
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_config,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
        client_selector=client_selectors.Random(1),
        trainers_data_dict=client_data,
        train_ratio=0.7,
        initial_model=lambda: LogisticRegression(41, 5),
        num_rounds=3,
        desired_accuracy=0.99,
        Selector=IOTLIST,
        DataName="kdd"
    )

    federated.add_subscriber(FederatedLogger([Events.ET_ROUND_FINISHED, Events.ET_TRAINER_SELECTED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    # federated.add_subscriber(fed_plots.RoundAccuracy(plot_ratio=0))
    # federated.add_subscriber(SQLiteLogger('avg_2'))

    logger.info("----------------------")
    logger.info("start federated 1")
    logger.info("----------------------")
    federated.start()

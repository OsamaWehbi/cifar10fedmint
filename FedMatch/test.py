import logging
import random
import time
import xlsxwriter
import Generate
import Preference
import Match
from src.data.data_distributor import LabelDistributor
from src.data.data_distributor import SizeDistributor
from src.data.data_loader import preload
from apps.experiments import federated_averaging as tfed
from apps.experiments import MyFunctions as MYF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test')
logger.info('Start Test\'s....')

# generate should take a dictionary of needed server with the number of clients needed by each
# plus the total number of iot devices
# save the data Partion for each data set
alldatapart = {}

# generate Iot and Fedservers
IotArr, FedArr, pardic = Generate.generate(4)

# print the FedServers
logger.info('Fed elements:')
for i in FedArr:
    logger.info('%s', i)

# prepare data dict for each IOT
for i, val in pardic.items():
    if i == 'kdd':
        # dist = SizeDistributor(num_clients=val)
        dist = LabelDistributor(num_clients=val, label_per_client=2, is_random_label_size=True)
        client_data = preload(i, dist)
        alldatapart[i] = client_data
    else:
        # dist = SizeDistributor(num_clients=val)
        dist = LabelDistributor(num_clients=val, label_per_client=4 , is_random_label_size=True)
        client_data = preload(i, dist)
        alldatapart[i] = client_data

# for t in alldatapart["kdd"].keys():
#     print(f"{t}")
# quit()
for j, val in alldatapart.items():
    if j == "mnist":
        tfed.mnistdata(IOTLIST=pardic[j], is_Dummy=True, dataset=val)
    else:
        tfed.kdddata(IOTLIST=pardic[j], is_Dummy=True, dataset=val)

logger.info('IoT elements:')
for i in IotArr:
    logger.info('%s', i)
# quit()
# tfed.mnistdata(IOTLIST=IotArr, is_Dummy=True)

# print (alldatapart['mnist'])
# quit()
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
IoTs_matching, Servers_m, Servers_matching = Match.federated_matching(IotArr, FedArr)

# logger.info('Matching Results:')
# for i, j in Servers_matching.items():
#     logger.info('%s : %s', i, j)
workbook = xlsxwriter.Workbook("test1.xlsx")
TFedArr = []

for i in FedArr:
    if i.getDName() == "mnist":
        TFedArr.append(MYF.mnistFed(FedServer=i, client_data=alldatapart[i.getDName()], tag='ICSF', workbook=workbook))
    else:
        TFedArr.append(MYF.KDDFED(FedServer=i, client_data=alldatapart[i.getDName()], tag='ICSF', workbook=workbook))

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
        MAX_Round =t.round
MAX_Round
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
        IoTs_matching, Servers_m, Servers_matching = Match.federated_matching(IotArr, FedArr)

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

for i in FedArr:
    i.round = i.mrounds

random.shuffle(IotArr)
random.shuffle(FedArr)

Preference.FedPreference(IotArr, FedArr)
Preference.IoTPreference(IotArr, FedArr)

IoTs_matching, Servers_m, Servers_matching = Preference.selectrandomiot(IotArr, FedArr)
# print(len(IotArr))
# print(len(FedArr))
# quit()
logger.info('Random Selection Results:')
for i, j in Servers_matching.items():
    logger.info('%s : %s', i, j)

TFedArr = []
for i in FedArr:
    if i.getDName() == "mnist":
        TFedArr.append(MYF.mnistFed(FedServer=i, client_data=alldatapart[i.getDName()], tag='Random', workbook=workbook))
    else:
        TFedArr.append(MYF.KDDFED(FedServer=i, client_data=alldatapart[i.getDName()], tag='Random', workbook=workbook))

for i in TFedArr:
    i.one_round()
    i.FedServer.round = i.FedServer.round - 1
# for i in IotArr:
#     i.FSweightdic = {}
#     i.preference = []
MAX_Round = -1
for t in FedArr:
    if t.round > MAX_Round:
        MAX_Round =t.round
MAX_Round
for count in range(MAX_Round):
        # add to preference of Fed
        random.shuffle(IotArr)
        random.shuffle(FedArr)
        Preference.FedPreference(IotArr, FedArr)
        # choose the method 2 criterion (Sort) OR MCDM (calculateSw)
        # Preference.calculateSwFed(FedArr)
        # Preference.sortFed(FedArr)
        # add to preference of IOT
        Preference.IoTPreference(IotArr, FedArr)
        # choose the method 2 criterion (Sort) OR MCDM (calculateSw)
        # Preference.calculateSwIot(IotArr)
        # Preference.sortIot(IotArr)
        # pass the iot arr and fed to match as is then create function in each class iot and arr that return
        # as result the name of the server with its preference same for iot...
        # receive them put them in dictionaries run match then
        # return as result 2 dictionaries and set the selected to preference if the FedServer
        # call match
        IoTs_matching, Servers_m, Servers_matching = Preference.selectrandomiot(IotArr, FedArr)

        logger.info('Random Selection Results:')
        for i, j in Servers_matching.items():
            logger.info('%s : %s', i, j)

        for i in TFedArr:
            i.one_round()
            i.FedServer.round = i.FedServer.round - 1
        # for i in IotArr:
        #     i.FSweightdic = {}
        #     i.preference = []
        # if count == MAX_Round:
        #     time.sleep(0.2)

workbook.close()
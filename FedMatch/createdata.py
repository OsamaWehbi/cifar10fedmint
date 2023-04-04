import logging
import random
import time
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import xlsxwriter
import Generate
import Preference
import Match
import numpy as nu
from src.data.data_distributor import LabelDistributor
from src.data.data_distributor import SizeDistributor
from src.data.data_loader import preload
from apps.experiments import federated_averaging as tfed
from apps.experiments import MyFunctions as MYF
import MLM
import pickle, os
addt = 10
def addIots(NIO=addt, oldArr=None, client_data=None):
    IotArr, pardic = Generate.generateIoT(NIO=NIO, IoTArr=oldArr, pAcc=False)
    # dist = SizeDistributor(num_clients=pardic)
    dist = LabelDistributor(num_clients=pardic, label_per_client=4, is_random_label_size=True)
    client_datas = preload('mnist', dist)
    # for x, res in client_datas.items():
    #     for iot in IotArr:
    #         if iot.Name == x:
    #             client_datas[x] = res.shuffle().as_tensor().poismain(IoT=iot, datatype='mnist')
    client_data = client_data.concat(client_datas)
    NPIoT = []
    DNIoT = {}
    Accmodel,Epr, Ere, Ede = MLM.bootModel()
    for l in IotArr:
        if sum(l.part_time.values()) == 0:
            producer = Epr.transform([l.Producer])
            region = Ere.transform([l.Region])
            dev = Ede.transform([l.Devt])
            arr = np.reshape([producer, region, dev], (1, 3))
            l.Dicaccuracy['mnist'] = Accmodel.predict(arr)[0]
            # l.Dicaccuracy['mnist'] = random.uniform(0,1)
            NPIoT.append(l)
            DNIoT[l.Name] = client_data[l.Name]
    tfed.mnistdata(IOTLIST=NPIoT, is_Dummy=True, dataset=DNIoT)
    return IotArr, client_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test')
logger.info('Start Test\'s....')

# generate Iot and Fedservers
IotArr, FedArr, pardic = Generate.generate(NOF=2, NIO=100)
dist = LabelDistributor(num_clients=pardic, label_per_client=4, is_random_label_size=True)
# dist = SizeDistributor(num_clients=pardic)
client_data = preload('mnist', dist)

# for x, y in client_data.items():
#     for iot in IotArr:
#         if iot.Name == x:
#             client_data[x] = y.shuffle().as_tensor().poismain(IoT=iot, datatype='mnist')

# prepare data dict for each IOT
# dist = SizeDistributor(num_clients=pardic)

tfed.mnistdata(IOTLIST=IotArr, is_Dummy=True, dataset=client_data)
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

IoTs_matching, Servers_m, Servers_matching, revenuedic = Match.federated_matching(IotArr, FedArr)

# logger.info('Fed elements:')
# for i in FedArr:
#     logger.info('%s', i)
#
# for i in IotArr:
#     logger.info('%s', i)
# quit()


# IotArr, client_data = addIots(NIO=50, oldArr=IotArr, client_data=client_data)
# for f in FedArr:
#     f.Export_data()
# MLM.updatebooModel()


workbook = xlsxwriter.Workbook("test1.xlsx")
TFedArr = []

stop=True
looper = True

for i in FedArr:
    TFedArr.append(MYF.mnistFed(FedServer=i, client_data=client_data, tag='ICSF', workbook=workbook))
# save the average revenue per round
reves = {}
# avrevli = {i.getName(): [] for i in FedArr if i.getName() not in avrevli.key()}
for i in TFedArr:
    i.one_round()
    reves[i.context.round_id] = revenuedic
    stop = stop and i.FedServer.finish
if stop:
    looper= False

while looper:

    for f in FedArr:
        f.Export_data()
    MLM.updatebooModel()
    Preference.UpdateIoT(IotArr, FedArr)
    IotArr, client_data = addIots(oldArr=IotArr, client_data=client_data)

    for tf in TFedArr:
        tf.temp_trainers_data_dict = client_data

    # for x in TFedArr:
    #     # if x.context.round_id>1 and x.context.round_id%10 == 1:
    #     tfed.kdddata(IOTLIST=IotArr, is_Dummy=True, dataset=client_data)
    stop = True
    # add to preference of Fed
    # Preference.UpdateIoT(IotArr, FedArr)
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
    IoTs_matching, Servers_m, Servers_matching, temprevenuedic = Match.federated_matching(IotArr, FedArr)
    revenuedic = dict(Counter(revenuedic) + Counter(temprevenuedic))
    # logger.info('Matching Results:')
    # for i, j in Servers_matching.items():
    #     logger.info('%s : %s', i, j)

    for i in TFedArr:
        i.one_round()
        # print(f"{stop} and  {not i.FedServer.finish}")
        stop = stop and i.FedServer.finish
        reves[i.context.round_id] = temprevenuedic
        # time.sleep(2)
    # print(looper)
    # time.sleep(2)
    if stop:
        # break
        looper= False
    # for i in IotArr:
    #     i.FSweightdic = {}
    #     i.preference = []
    # if count == MAX_Round:
    #     time.sleep(0.2)
    # logger.info('IoT elements:')
    # for i in IotArr:
    #     logger.info('%s', i)
# logger.info('*********************************')
# logger.info('*********************************')
# logger.info('*********************************')
# logger.info('*********************************')
# quit()


for i in FedArr:
    i.finish = False
case = ['ICSF', 'random']
data = {}
rev = []
for i in IotArr:
    if i.crev != 0:
        rev.append(i.crev)
        i.setcrev(0)
data["revenue"] = [sum(rev)/len(rev)]
data["number of clients"] = [len(rev)]
print("************************************************")
print(f"{sum(rev)} and {len(rev)}")
print(sum(rev)/len(rev))
print("************************************************")

time.sleep(1.0)
IotArr.sort(key=lambda x: x.ID)
print(len(IotArr))
logger.info('IoT elements:')
for i in IotArr:
    logger.info('%s', i)
rIotArr = IotArr
bound =100

ubound = addt

IotArr = rIotArr[:bound]

# workbook.close()
print(MLM.resarr)
print(MLM.resarr.values())

quit()



Preference.UpdateIoT(IotArr, FedArr)
random.shuffle(IotArr)
random.shuffle(FedArr)
# neeed to check it is changing not same devices any more


Preference.RFedPreference(IotArr, FedArr)
Preference.RIoTPreference(IotArr, FedArr)

IoTs_matching, Servers_m, Servers_matching, rreven = Preference.selectrandomiot(IotArr, FedArr)
# print(len(IotArr))
# print(len(FedArr))
# quit()
logger.info('Random Selection Results:')
for i, j in Servers_matching.items():
    logger.info('%s : %s', i, j)

TFedArr = []
for i in FedArr:
    TFedArr.append(MYF.mnistFed(FedServer=i, client_data=client_data, tag='Random', workbook=workbook))

stop=True
looper = True
rrevres ={}
for i in TFedArr:
    i.one_round()
    stop = stop and i.FedServer.finish
    rrevres[i.context.round_id] = rreven
if stop:
    looper= False


while looper:

        bound += ubound
        IotArr = rIotArr[:bound]
        stop = True
        # add to preference of Fed
        Preference.UpdateIoT(IotArr, FedArr)
        random.shuffle(IotArr)
        random.shuffle(FedArr)


        Preference.RFedPreference(IotArr, FedArr)
        # choose the method 2 criterion (Sort) OR MCDM (calculateSw)
        # Preference.calculateSwFed(FedArr)
        # Preference.sortFed(FedArr)
        # add to preference of IOT
        Preference.RIoTPreference(IotArr, FedArr)
        # choose the method 2 criterion (Sort) OR MCDM (calculateSw)
        # Preference.calculateSwIot(IotArr)
        # Preference.sortIot(IotArr)
        # pass the iot arr and fed to match as is then create function in each class iot and arr that return
        # as result the name of the server with its preference same for iot...
        # receive them put them in dictionaries run match then
        # return as result 2 dictionaries and set the selected to preference if the FedServer
        # call match
        IoTs_matching, Servers_m, Servers_matching, temprrev = Preference.selectrandomiot(IotArr, FedArr)
        rreven = dict(Counter(rreven)+Counter(temprrev))
        logger.info('Random Selection Results:')
        for i, j in Servers_matching.items():
            logger.info('%s : %s', i, j)

        for i in TFedArr:
            i.one_round()
            stop = stop and i.FedServer.finish
            rrevres[i.context.round_id] = temprrev
        if stop:
            # break
            looper=False
        # for i in IotArr:
        #     i.FSweightdic = {}
        #     i.preference = []
        # if count == MAX_Round:
        #     time.sleep(0.2)

# logger.info('IoT elements:')
# for i in IotArr:
#     logger.info('%s', i)
rev = []
for i in IotArr:
    if i.crev != 0:
        rev.append(i.crev)
        # i.setcrev(0)
data["revenue"].append(sum(rev)/len(rev))
data["number of clients"].append(len(rev))
print("************************************************")
print(f"{sum(rev)} and {len(rev)}")
print(sum(rev)/len(rev))
print("************************************************")
df = pd.DataFrame(data, index=case)
print(df)
print(revenuedic)
print(rreven)
print(reves)
print(rrevres)
# Ygirls = [10, 20, 20, 40]
# Zboys = [20, 30, 25, 30]
#
# X_axis = np.arange(len(case))
#
# plt.bar(X_axis - 0.2, Ygirls, 0.4, label='Girls')
# plt.bar(X_axis + 0.2, Zboys, 0.4, label='Boys')
#
# plt.xticks(X_axis, case)
# plt.xlabel("Groups")
# plt.ylabel("Number of Students")
# plt.title("Number of Students in each group")
# plt.legend()
# plt.show()
worksheet = workbook.add_worksheet('Price')
worksheet.write('A1', 'Round#')
worksheet.write('B1', 'Fed1 ICSF')
worksheet.write('C1', 'Fed1 Random')
worksheet.write('D1', 'Fed2 ICSF')
worksheet.write('E1', 'Fed2 Random')

for i, val in reves.items():
    # res = list(val.values())
    worksheet.write('A' + str(i + 1), i)
    for x, y in val.items():
        if x == 'Fed1':
            worksheet.write('B' + str(i + 1), y)
        else:
            worksheet.write('D' + str(i + 1), y)

for i, val in rrevres.items():
    for x, y in val.items():
        if x == 'Fed1':
            worksheet.write('C' + str(i + 1), y)
        else:
            worksheet.write('E' + str(i + 1), y)
    data["revenue"].append(sum(rev) / len(rev))
    data["number of clients"].append(len(rev))
i=2
worksheet.write('H1', "ICSF")
worksheet.write('I1', "Random")
worksheet.write('G2', "Revenue")
worksheet.write('G3', "nb clients participated")
for x, y in data.items():
    worksheet.write('H' + str(i), y[0])
    worksheet.write('I' + str(i), y[1])
    i +=1
print(MLM.resarr)

# IotArr.sort(key=lambda x: x.ID)
# for i in IotArr:
#     logger.info('%s', i)
# for i in IotArr:
#     logger.info('%s', i)
workbook.close()


import random
import time

import FedServer
import IoT
import logging
import xlsxwriter
import MLM
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Generate')

#Loading Models
# Cmodel = MLM.CPUMODEL()
# Rmodel = MLM.RAMODEL()
Tmodel = MLM.TIMEMODEL()
# Emodel = MLM.ENMODEL()
Accmodel,Epr, Ere, Ede = MLM.bootModel()

# save the number of devices in each data set
# partdic = {}

# DataName = ["mnist", "mnist10k", "femnist", "kdd", "cifar10"]
# used to manage the number of Fedserver for each datatype (to be equal)
DataName = {}

# represent the resources of an IOT
RName = ["cpu", "ram", "bandwidth", "datasize"]
PName = ["cpu", "ram", "bandwidth"]
Region = ["America", "Asia", "Europe", "Africa"]
Producer = ["SAP","IBM","Cisco","PTC"]
Best = ["SAPAsiaPhone","SAPAfricaPhone","SAPAmericaWatch","SAPEuropeLock","IBMAsiaLock","IBMAfricaPhone","IBMAmericaSecurity","IBMEuropePhone","CiscoAsiaPhone","CiscoAfricaWatch","CiscoAmericaSecurity","CiscoEuropeWatch","PTCAsiaWatch","PTCAfricaSecurity","PTCAmericaSecurity","PTCEuropeLock"]
# DevType = ["camera", "watch", "car", "Phone", "fridge", "stove", "washing", " lock", "coffeemaker", "voicecontroller", "Airmonitor", "canary"]
DevType = ["Watch", "Phone", "Lock", "Security"]
# used to manage data size in general (used,wasted)
# , "kdd": 125000 "mnist": 55000,
Mdata = {"cifar10": 60000}
#
# Mdata = {"kdd": 125000}
# wr = xlsxwriter.Workbook("test2.xlsx")
# ws = wr.add_worksheet('tester')
# save generated federated servers
FedArr = []
# save generated IOT devices
# save the counter
IoTN =0
# dname = "kdd"
dname = "cifar10"

def generate(NOF=None, NIO=None):
    # print(Mdata)
    # quit()
    logger.info('Start Generate Fed Servers and IoT\'s....')
    global FedArr
    # IotArr = []
    GIoTNum = 100
    BIoTNum = 300
    # Fed_min_rounds = 20
    # Fed_max_rounds = 30
    Fed_min_rounds = 500
    Fed_max_rounds = 500
    # 30 50 / 5 10 / 20 30
    fed_min_iot = 10
    fed_max_iot = 10
    close_min_cpu_price = 0.001
    close_max_cpu_price = 0.009
    fed_min_price = 0.001
    fed_max_price = 0.009
    # fed_min_ram_price = 400
    # fed_max_ram_price = 900
    # fed_min_DataSize_price = 100
    # fed_max_DataSize_price = 3000
    # fed_min_bandwidth_price = 500
    # fed_max_bandwidth_price = 900
    if NOF:
        # create Fed
        # NumPerDev = int(NOF / len(Mdata.keys()))
        # Remain = int(((NOF / len(Mdata.keys())) - NumPerDev) * len(Mdata.keys()))
        # DataName = {h: NumPerDev for h in Mdata.keys()}
        # DataName[random.choice(list(Mdata))] += Remain
        for i in range(NOF):
            # qex = False
            # while not qex:
                # pic random dataName
            #     DName = random.choice(list(DataName.keys()))
            #     if DataName[DName] > 0:
            #         qex = True
            # DataName[DName] -= 1
            DName = dname
            # generate random number of IoT's
            num = random.randint(fed_min_iot, fed_max_iot)
            # IoTNum += num
            # create price dictionary
            # price_dct = {PName[c]: format(random.uniform(close_min_cpu_price, close_max_cpu_price), ".6f") for c in
            #              range(0, len(PName))}
            # generate random number of rounds
            rounds = random.randint(Fed_min_rounds, Fed_max_rounds)
            # create the Fedserver Device
            Fed_device_Match = FedServer.FedServer(i + 1, "Fed" + str(i + 1), DName, num, rounds)
            FedArr.append(Fed_device_Match)
        IotArr,  partdic = generateIoT(NIO=NIO)
    return IotArr, FedArr, partdic
def generateIoT(NIO=None, IoTArr=None, pAcc=True):
    IotArr = []
    if IoTArr:
        IotArr = IoTArr
    ###############################################################
    ############### Generate IoT's ###############################
    ###############################################################
    # prepare the total number of IoT's based on the required:
    # oldNum = IoTNum
    # for testing
    # IoTNum = 250
    # IoTNum += 5
    # Store the IoT devices Generated
    partdic = []
    # set the minimum and maximum resources units of the IoT
    iot_min_cpu = 300  # 200
    iot_max_cpu = 700  # 300
    bad_iot_min_cpu = 300  # 50
    bad_iot_max_cpu = 320  # 100
    iot_min_ram = 400  # 100
    iot_max_ram = 900  # 200
    bad_iot_min_ram = 400  # 20
    bad_iot_max_ram = 420  # 30
    iot_min_bandwidth = 500  # 300
    iot_max_bandwidth = 900  # 400
    bad_iot_min_bandwidth = 500  # 15
    bad_iot_max_bandwidth = 520  # 20
    iot_min_Latency = 100
    iot_max_Latency = 500
    # 100 200 80 100 / 200 200 100 100
    iot_min_DataSize = 300  # 100
    iot_max_DataSize = 450  # 1000
    bad_iot_min_DataSize = 100
    bad_iot_max_DataSize = 100
    far_Fed_min_Latency = 0.5
    far_Fed_max_Latency = 1
    close_Fed_min_Latency = 0.001
    close_Fed_max_Latency = 0.01
    far_min_cpu_price = 0.0001
    far_max_cpu_price = 0.0009
    close_min_cpu_price = 0.001
    close_max_cpu_price = 0.009

    datas = 50
    global IoTN
    iend = NIO+IoTN
    for i in range(IoTN, iend):

        lst = list(Mdata.keys())
        # data_dct = {lst[i]: datas for i in range(0, len(lst))}
        # data_dct = {dname: random.randint(500, 1500)}
        diclatency = {}
        ipricedic = {}
        region = random.choice(Region)
        devt = random.choice(DevType)
        producer = random.choice(Producer)
        # cpu = float(Cmodel.predict(xdata))
        # ram = float(Rmodel.predict(xdata))

        # bandwith = random.randint(iot_min_bandwidth, iot_max_bandwidth)
        # print(len(DataName))
        # quit()
        # lst = random.sample(list(DataName), random.randint(1, len(DataName)))
        # random.seed(50)

        # datas += 50
        # for ttype, val in data_dct.items():
        #     if ttype in Mdata.keys():
        #         Mdata[ttype] -= val

        for s in FedArr:
            if s.getDName() in lst:
                latency = round(random.uniform(close_Fed_min_Latency, far_Fed_max_Latency), 4)
                if (producer+region+devt) in Best:
                    # , "Americawatch", "AsiaPhone", "Europewatch", "Africalock"
                    price = {PName[c]: format(random.uniform(close_min_cpu_price, close_max_cpu_price), ".6f") for c in
                             range(0, len(PName))}
                    # cpu += random.randint(20,50)
                    # ram += random.randint(20,50)
                else:
                    price = {PName[c]: format(random.uniform(far_min_cpu_price, far_max_cpu_price), ".6f") for c in
                         range(0, len(PName))}
                # print(latency)
                ipricedic[s.getName()] = price
                diclatency[s.getName()] = latency

        if (producer+region+devt) in Best:
            data_dct = {dname: random.randint(300, 450)}
            cpu = random.randint(iot_min_cpu, iot_max_cpu)
            ram = random.randint(iot_min_ram, iot_max_ram)
            bandwith = random.randint(iot_min_bandwidth, iot_max_bandwidth)
            # data_dct = {dname: random.randint(900, 1500)}
        else:
            data_dct = {dname: random.randint(100, 300)}
            cpu = random.randint(bad_iot_min_cpu, bad_iot_max_cpu)
            ram = random.randint(bad_iot_min_ram, bad_iot_max_ram)
            bandwith = random.randint(bad_iot_min_bandwidth, bad_iot_max_bandwidth)

        xdata = np.array(data_dct[dname]).reshape(-1, 1)
        # iotTaskTimeNeededForTheJob = float(Tmodel.predict(xdata))
        iotTaskTimeNeededForTheJob = float(100)

        iot_device_Match = IoT.IoT(i + 1, "IOT" + str(i + 1), cpu, ram, bandwith,
                                   iotTaskTimeNeededForTheJob, data_dct, diclatency, ipricedic, region, devt, producer)
        # ws.write('A' + str(i+1), "IOT" + str(i + 1))
        # ws.write('B' + str(i+1), producer)
        # ws.write('C' + str(i+1), region)
        # ws.write('D' + str(i+1), devt)
        # ws.write('E' + str(i+1), data_dct['mnist'])

        # ws.write('F' + str(i+1), ram)
        for f in FedArr:
            iot_device_Match.part_time[f.Name] = 0
        if pAcc:
            producer = Epr.transform([producer])
            region = Ere.transform([region])
            dev = Ede.transform([devt])
            arr = np.reshape([producer, region, dev], (1, 3))
            iot_device_Match.Dicaccuracy[dname] = Accmodel.predict(arr)[0]
            # iot_device_Match.Dicaccuracy[dname] = random.uniform(0, 1)
        else:
            iot_device_Match.Dicaccuracy[dname] = -1
        IotArr.append(iot_device_Match)

        partdic.append(iot_device_Match)

    # IoTNum += random.randint(10, 20)
    IoTN = i+1
    # maxss = int(IoTNum + 150)
    # wr.close()
    return IotArr, partdic

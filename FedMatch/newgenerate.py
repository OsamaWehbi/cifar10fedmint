import random
import FedServer
import IoT
import logging
import MLM
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Generate')
# Tmodel = MLM.TIMEMODEL()
Accmodel, Epr, Ere, Ede = MLM.bootModel()

# DataName = ["mnist", "mnist10k", "femnist", "kdd", "cifar10"]
# used to manage the number of Fedserver for each datatype (to be equal)
DataName = {}

# represent the resources of an IOT
RName = ["cpu", "ram", "bandwidth", "datasize"]
PName = ["cpu", "ram", "bandwidth"]
Region = ["America", "Asia", "Europe", "Africa"]
Producer = ["SAP", "IBM", "Cisco", "PTC"]
Best = ["SAPAsiaPhone", "SAPAfricaPhone", "SAPAmericaWatch", "SAPEuropeLock", "IBMAsiaLock", "IBMAfricaPhone",
        "IBMAmericaSecurity", "IBMEuropePhone", "CiscoAsiaPhone", "CiscoAfricaWatch", "CiscoAmericaSecurity",
        "CiscoEuropeWatch", "PTCAsiaWatch", "PTCAfricaSecurity", "PTCAmericaSecurity", "PTCEuropeLock"]
# DevType = ["camera", "watch", "car", "Phone", "fridge", "stove", "washing", " lock", "coffeemaker", "voicecontroller", "Airmonitor", "canary"]
DevType = ["Watch", "Phone", "Lock", "Security"]
# used to manage data size in general (used,wasted)
Mdata = {"cifar10": 60000}
# save generated IOT devices
# save the counter
dname = "cifar10"


def generate(NOF=None, NIO=None):
    logger.info('Start Generate Fed Servers and IoT\'s....')
    FedArr = []
    IotArr = []
    partdic = []
    GIoTNum = 70
    BIoTNum = 29
    Fed_max_rounds = 500
    fed_max_iot = 10
    bad_iot_min_cpu = 300  # 50
    bad_iot_max_cpu = 320  # 100
    bad_iot_min_ram = 400  # 20
    bad_iot_max_ram = 420  # 30
    bad_iot_min_bandwidth = 500  # 15
    bad_iot_max_bandwidth = 520  # 20
    far_Fed_max_Latency = 1
    close_Fed_min_Latency = 0.001
    far_min_cpu_price = 0.0001
    far_max_cpu_price = 0.0009
    close_min_cpu_price = 0.001
    close_max_cpu_price = 0.009
    if NOF:
        # create Fed
        for i in range(NOF):
            # create the Fedserver Device
            Fed_device_Match = FedServer.FedServer(i + 1, "Fed" + str(i + 1), dname, fed_max_iot, Fed_max_rounds)
            FedArr.append(Fed_device_Match)

    # GENERATE BAD IOT
    for i in range(BIoTNum):
        diclatency = {}
        ipricedic = {}
        region = random.choice(Region)
        devt = random.choice(DevType)
        producer = random.choice(Producer)
        while (producer + region + devt) in Best:
            region = random.choice(Region)
            devt = random.choice(DevType)
            producer = random.choice(Producer)
        for s in FedArr:
            latency = round(random.uniform(close_Fed_min_Latency, far_Fed_max_Latency), 4)
            price = {PName[c]: format(random.uniform(far_min_cpu_price, far_max_cpu_price), ".6f") for c in
                     range(0, len(PName))}
            ipricedic[s.getName()] = price
            diclatency[s.getName()] = latency
        data_dct = {dname: random.randint(100, 300)}
        cpu = random.randint(bad_iot_min_cpu, bad_iot_max_cpu)
        ram = random.randint(bad_iot_min_ram, bad_iot_max_ram)
        bandwith = random.randint(bad_iot_min_bandwidth, bad_iot_max_bandwidth)
        iotTaskTimeNeededForTheJob = float(100)
        iot_device_Match = IoT.IoT(i, "IOT" + str(i), cpu, ram, bandwith,
                                   iotTaskTimeNeededForTheJob, data_dct, diclatency, ipricedic, region, devt, producer)
        producer = Epr.transform([producer])
        region = Ere.transform([region])
        dev = Ede.transform([devt])
        arr = np.reshape([producer, region, dev], (1, 3))
        iot_device_Match.setAcc(Accmodel.predict(arr)[0], dname)
        for f in FedArr:
            iot_device_Match.part_time[f.Name] = 0
        IotArr.append(iot_device_Match)
        partdic.append(iot_device_Match)

    for i in range(BIoTNum, GIoTNum+BIoTNum):
        diclatency = {}
        ipricedic = {}
        region = random.choice(Region)
        devt = random.choice(DevType)
        producer = random.choice(Producer)
        while (producer + region + devt) not in Best:
            region = random.choice(Region)
            devt = random.choice(DevType)
            producer = random.choice(Producer)
        for s in FedArr:
            latency = round(random.uniform(close_Fed_min_Latency, far_Fed_max_Latency), 4)
            price = {PName[c]: format(random.uniform(close_min_cpu_price, close_max_cpu_price), ".6f") for c in
                     range(0, len(PName))}
            ipricedic[s.getName()] = price
            diclatency[s.getName()] = latency
        data_dct = {dname: random.randint(600, 600)}
        cpu = random.randint(bad_iot_min_cpu, bad_iot_max_cpu)
        ram = random.randint(bad_iot_min_ram, bad_iot_max_ram)
        bandwith = random.randint(bad_iot_min_bandwidth, bad_iot_max_bandwidth)
        iotTaskTimeNeededForTheJob = float(100)
        iot_device_Match = IoT.IoT(i, "IOT" + str(i), cpu, ram, bandwith,
                                   iotTaskTimeNeededForTheJob, data_dct, diclatency, ipricedic, region, devt, producer)
        producer = Epr.transform([producer])
        region = Ere.transform([region])
        dev = Ede.transform([devt])
        arr = np.reshape([producer, region, dev], (1, 3))
        iot_device_Match.setAcc(Accmodel.predict(arr)[0], dname)
        for f in FedArr:
            iot_device_Match.part_time[f.Name] = 0
        IotArr.append(iot_device_Match)
        partdic.append(iot_device_Match)

    return IotArr, FedArr, partdic

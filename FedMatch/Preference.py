# theory of MCDA multiple criteria decision analysis beneficial and non beneficial
import logging
import random
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preference')

def UpdateIoT(IoTArr, FedArr):
    # IoTArr.sort(key=lambda x: x.ID)
    PName = ["cpu", "ram", "bandwidth"]
    Best = ["SAPAsiaPhone", "SAPAfricaPhone", "SAPAmericaWatch", "SAPEuropeLock", "IBMAsiaLock", "IBMAfricaPhone",
            "IBMAmericaSecurity", "IBMEuropePhone", "CiscoAsiaPhone", "CiscoAfricaWatch", "CiscoAmericaSecurity",
            "CiscoEuropeWatch", "PTCAsiaWatch", "PTCAfricaSecurity", "PTCAmericaSecurity", "PTCEuropeLock"]
    GIoTNum = 100
    BIoTNum = 300
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
    close_min_cpu_price = 0.001
    close_max_cpu_price = 0.009
    far_min_cpu_price = 0.0001
    far_max_cpu_price = 0.0009

    iot_min_Latency = 100
    iot_max_Latency = 500
    # 100 200 80 100 / 200 200 100 100
    iot_min_DataSize = 300  # 100
    iot_max_DataSize = 400  # 1000
    bad_iot_min_DataSize = 100
    bad_iot_max_DataSize = 100
    far_Fed_min_Latency = 0.5
    far_Fed_max_Latency = 1
    close_Fed_min_Latency = 0.001
    close_Fed_max_Latency = 0.01
    for i in IoTArr:
        for s in FedArr:
            latency = round(random.uniform(close_Fed_min_Latency, far_Fed_max_Latency), 4)
            if (i.Producer+i.Region+i.Devt) in Best:
                # , "Americawatch", "AsiaPhone", "Europewatch", "Africalock"
                price = {PName[c]: format(random.uniform(close_min_cpu_price, close_max_cpu_price), ".6f") for c in
                         range(0, len(PName))}
            else:
                price = {PName[c]: format(random.uniform(far_min_cpu_price, far_max_cpu_price), ".6f") for c in
                     range(0, len(PName))}
            # print(latency)
            i.pricedic[s.Name] = price
            i.diclatency[s.Name] = latency

        if (i.Producer+i.Region+i.Devt) in Best:
            cpu = random.randint(iot_min_cpu, iot_max_cpu)
            ram = random.randint(iot_min_ram, iot_max_ram)
            bandwidth = random.randint(iot_min_bandwidth, iot_max_bandwidth)
        else:
            cpu = random.randint(bad_iot_min_cpu, bad_iot_max_cpu)
            ram = random.randint(bad_iot_min_ram, bad_iot_max_ram)
            bandwidth = random.randint(bad_iot_min_bandwidth, bad_iot_max_bandwidth)
        i.cpu = cpu
        i.ram = ram
        i.bandwidth = bandwidth

def FedPreference(IotArr, FedArr):
    logger.info("Add IoT's To Preference of Fed....")
    for i in FedArr:
        i.preference = []
        # if i.round > 0 and (i.finish == False):
        if i.finish == False:
            for j in IotArr:
                if i.getDName().lower() in j.getData().keys() and j.part_time[i.Name] < 3:
                # if i.getDName().lower() in j.getData().keys():
                    # i.preference.append(copy.deepcopy(j))
                    i.preference.append(j)
                else:
                    j.FSweightdic = {}
    logger.info("End Add IoT's To Preference of Fed....")
def RFedPreference(IotArr, FedArr):
    logger.info("Add IoT's To Preference of Fed....")
    for i in FedArr:
        i.preference = []
        # if i.round > 0 and (i.finish == False):
        if i.finish == False:
            for j in IotArr:
                # if i.getDName().lower() in j.getData().keys() and j.part_time[i.Name] < 4:
                if i.getDName().lower() in j.getData().keys():
                    # i.preference.append(copy.deepcopy(j))
                    i.preference.append(j)
                else:
                    j.FSweightdic = {}
    logger.info("End Add IoT's To Preference of Fed....")

def IoTPreference(IotArr, FedArr):
    logger.info("Add Fed's To Preference of IoT....")
    for i in IotArr:
        i.preference = []
        for j in FedArr:
            # if j.round > 0 and not j.finish:
            if j.finish == False:
                # if j.getDName().lower() in i.getData().keys() :
                if j.getDName().lower() in i.getData().keys() and i.part_time[j.Name] < 3:
                    # i.arraypreference.append(copy.deepcopy(j))
                    i.preference.append(j)
    logger.info("End Add Fed's To Preference of IoT....")

def RIoTPreference(IotArr, FedArr):
    logger.info("Add Fed's To Preference of IoT....")
    for i in IotArr:
        i.preference = []
        for j in FedArr:
            # if j.round > 0 and not j.finish:
            if j.finish == False:
                if j.getDName().lower() in i.getData().keys() :
                # if j.getDName().lower() in i.getData().keys() and i.part_time[j.Name] < 4:
                    # i.arraypreference.append(copy.deepcopy(j))
                    i.preference.append(j)
    logger.info("End Add Fed's To Preference of IoT....")

# print(IotArr[1])
def calculateSwFed(FedArr):
    logger.info("Start calculate IoT's Preference In Fed....")
    for i in FedArr:
        i.FedSw()
    logger.info("End calculate IoT's Preference In Fed....")
    # for i in FedArr:
    #     print(f" for {i.Name} res {i.ISweightdic}")


def calculateSwIot(IotArr):
    logger.info("Start calculate Fed's Preference In IoT....")
    for j in IotArr:
        j.IoTSW()
    logger.info("End calculate Fed's Preference In IoT....")


def sortFed(FedArr):
    logger.info("Start Sorting IoT's Preference In Fed....")
    for i in FedArr:
        i.FedSort()
    logger.info("End calculate IoT's Preference In Fed....")


def sortIot(IotArr):
    logger.info("Start Sorting Fed's Preference In IoT....")
    for j in IotArr:
        j.IoTSort()
    logger.info("End Sorting Fed's Preference In IoT....")

def selectrandomiot(IotArr, FedArr):
    IoTs_picks = {}
    Servers_picks = {}
    Servers_NumClients = {}
    for i in IotArr:
        for j in i.preference:
            if i.getName() in IoTs_picks.keys():
                IoTs_picks[i.getName()].append(j.getName())
            else:
                IoTs_picks[i.getName()] = [j.getName()]

        # print(f"{i.FSweightdic}")
    for i in FedArr:
        # if i.round > 0 and not i.finish:
        if i.finish == False:
            for j in i.preference:
                if i.getName() in Servers_picks.keys():
                    Servers_picks[i.getName()].append(j.getName())
                else:
                    Servers_picks[i.getName()] = [j.getName()]
            # NumClients[i.getName()] = i.IOTNum
            Servers_NumClients[i.getName()] = i.IOTNum
            # print(i.getName())
            # for y in i.preference:
            #     print(f"{y.getName()} ==> {y.getAcc()}")
    # quit()
    # logger.info('IoT preferences dictionary :')
    # for i, val in IoTs_picks.items():
    #     logger.info('%s : %s', i, val)
    # logger.info(IoTs_picks)
    logger.info('FedServer random dictionary :')
    for i, val in Servers_picks.items():
        logger.info('%s : %s', i, val)
    # print(Servers_picks)
    logger.info('Fed Servers Required IoT\'s :')
    logger.info('%s', Servers_NumClients)
    # print(Servers_NumClients)

    logger.info('Start Random Selection\'s....')
    # return list of IoT's
    IoTs = list(IoTs_picks.keys())
    # contain iots with their fed after matching
    IoTs_matching = {r: None for r in IoTs_picks.keys()}
    # contain Fed with list of their IOTs
    Servers_matching = {h: [] for h in Servers_picks.keys()}
    # print(IoTs_picks)
    # print(Servers_picks)
    # print(Servers_NumClients)
    # quit()
    try:
        while IoTs:
            # pick iot
            # r = IoTs.pop(random.randrange(len(IoTs)))
            r = IoTs.pop(0)
            # print (f" first {r}")
            # quit()

            # let r_p be the preferences of IoT r
            r_p = IoTs_picks[r]
            # print (r_p)
            # quit()
            # if r_p not empty and resident_matching(iot matching result fed1 fed 2..) is empty do ..
            while r_p and (not IoTs_matching[r]):
                # check if (iot) device exist in hi preference list of his preferable fed
                # r == one IoT
                # if r not in Servers_picks[r_p[0]]:
                #     r_p.remove(r_p[0])
                # if r-p iot preference not empty
                if r_p:
                    # pick random Fed from list
                    h = random.choice(r_p)
                    # let h_p be the preferences of Fedratedserver h
                    # get the preference list of the picked fed
                    #
                    # h_p = Servers_picks[h]
                    # let h_matches be the matched iot for Fedrated server h
                    # get the matched list of iot for the fed (should contain the iot devices after matching done)
                    h_matches = Servers_matching[h]
                    # print(h_matches)
                    # quit()
                    # if there is capacity add
                    if len(h_matches) < Servers_NumClients[h]:
                        IoTs_matching[r] = h
                        Servers_matching[h] += [r]
                    # if not
                    else:
                        r_p.remove(h)
                        IoTs.append(r)
                        # IoTs_picks[r].remove(h)
    except:
        print("out")
        traceback.print_exc()
    Servers_m = {}
    for i, val in Servers_matching.items():
        for j in FedArr:
            if i == j.getName():
                j.Mdic = val

    for i, val in Servers_matching.items():
        for j in val:
            for y in IotArr:
                if j == y.getName():
                    if i in Servers_m:
                        Servers_m[i].append(y)
                    else:
                        Servers_m[i] = [y]

    temprevenuedic = {h: [] for h in Servers_matching.keys()}
    revenuedic = {h: [] for h in Servers_matching.keys()}
    for i, val in Servers_m.items():
        for j in FedArr:
            if i == j.getName():
                j.preference = val
                for z in val:
                    z.addcrev(j)
                    temprevenuedic[j.getName()].append(z.rev[j.getName()])
    for i, val in temprevenuedic.items():
        revenuedic[i] = sum(list(val)) / len(val)

    logger.info('Random Selection Done\'s....')
    return IoTs_matching, Servers_m, Servers_matching, revenuedic

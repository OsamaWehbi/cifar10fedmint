import IoT
import logging
from csv import writer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FedServer')


class FedServer:
    """Class fed Device"""

    def __init__(self, *args):
        """initialize"""
        # i + 1, "Fed" + str(i + 1), DName, num, price_dct
        self.ID = args[0]
        self.Name = args[1]
        # the required data name for training
        self.DataName = args[2]
        # number of IoT's needed
        self.IOTNum = args[3]
        # rounds
        self.round = args[4]
        # for cpu,ram,bandwidth,data size...
        # self.prices = args[5].copy()
        # preference list of iot objects sorted by acc
        self.preference = []
        # self.model
        # save the finally selected
        # self.selected = []
        # task execution time
        self.fedtime = 0
        # federated server weight (or IOT's Weights which is better)
        self.ISweightdic = {}
        # matching res dic named of iots
        self.Mdic = []
        # backup rounds
        self.mrounds = args[4]
        #
        self.finish = False

    def getID(self):
        return self.ID

    def setID(self, iD):
        self.ID = iD

    def getName(self):
        return self.Name

    def setName(self, name):
        self.Name = name

    def getDName(self):
        return self.DataName

    def setDName(self, dName):
        self.DataName = dName

    def __str__(self):
        return f"FedServer ID = {self.ID} \tName = {self.Name} \tRound = {self.round} \tDataRequired = {self.DataName: <9} \t" \
               f"Number of Client = {self.IOTNum} \tPreference List = {self.preference}" \
               f" \tFederated Weight = {self.ISweightdic} \tfedTime = {self.fedtime}"

    # def getSelected(self):
    #     return self.selected
    #
    # def setSelected(self, selected):
    #     self.selected = selected

    def getfedtime(self):
        return self.fedtime

    def setfedtime(self, fedtime):
        self.fedtime = fedtime

    # def FedSw(self):
    #     self.ISweightdic = {}
    #     maxc = maxr = maxb = maxd = maxacc = 0
    #     minprice = minlat = 66666
    #     for i in self.preference:
    #         if maxc <= i.getCpu():
    #             maxc = i.getCpu()
    #
    #         if maxr <= i.getRam():
    #             maxr = i.getRam()
    #
    #         if maxb <= i.getBandwidth():
    #             maxb = i.getBandwidth()
    #
    #         if maxd <= int(i.getData()[self.DataName]):
    #             maxd = int(i.getData()[self.DataName])
    #
    #         if maxacc <= i.getAcc(self.DataName):
    #             maxacc = i.getAcc(self.DataName)
    #
    #         if minlat >= i.getFedlatency(self):
    #             minlat = i.getFedlatency(self)
    #
    #             price = (i.getCpu() * float(self.prices["cpu"])) + (i.getRam() * float(self.prices["ram"])) + (
    #                         i.getBandwidth() * float(self.prices["bandwidth"])) + (
    #                                 float(i.getData()[self.DataName]) * float(self.prices["datasize"]))
    #             # tr = self.prices["cpu"]
    #             # print(f"{type(price)} val of = {price}")
    #             # quit()
    #             if minprice >= price:
    #                 minprice = price
    #
    #     for i in self.preference:
    #         # print(f"acc {((i.getAcc() / maxacc) * 0.16) } cpu  {((i.getCpu() / maxc) * 0.14)} ram {((i.getRam() / maxr) * 0.14)}  band {((i.getBandwidth() / maxb) * 0.14)} data {((int(i.getData()[self.DataName]) / maxd) * 0.14)} lat {((minlat / i.getFedlatency(self)) * 0.14)} pr {((minprice / Iprice) * 0.14)}")
    #         # print(f" {ISweight}")
    #         # list1 = [maxc, maxr, maxb, maxd, maxacc, minlat, minprice]
    #         # list2 = [i.getAcc(), i.getCpu(), i.getRam(), i.getBandwidth(), i.getData()[self.DataName],
    #         #          i.getFedlatency(self), Iprice]
    #         # print(list1)
    #         # print(list2)
    #         Iprice = (i.getCpu() * float(self.prices["cpu"])) + (i.getRam() * float(self.prices["ram"])) + (
    #                     i.getBandwidth() * float(self.prices["bandwidth"])) + (
    #                              float(i.getData()[self.DataName]) * float(self.prices["datasize"]))
    #         ISweight = ((i.getAcc(self.DataName) / maxacc) * 0.16) + ((i.getCpu() / maxc) * 0.14) + ((i.getRam() / maxr) * 0.14) + (
    #                     (i.getBandwidth() / maxb) * 0.14) + ((float(i.getData()[self.DataName]) / maxd) * 0.14) + (
    #                                (minlat / i.getFedlatency(self)) * 0.14) + ((minprice / Iprice) * 0.14)
    #
    #         # quit()
    #         self.ISweightdic[i.getName()] = ISweight
    #
    #     self.ISweightdic = dict(sorted(self.ISweightdic.items(), key=lambda item: item[1], reverse=True))
    #     self.preference = sorted(self.preference, key=lambda x: self.ISweightdic[x.getName()], reverse=True)

    def FedSort(self):
        self.ISweightdic = {}
        for i in self.preference:
            self.ISweightdic[i.getName()] = i.getAcc(self.DataName)
        self.ISweightdic = dict(sorted(self.ISweightdic.items(), key=lambda item: item[1], reverse=True))
        self.preference.sort(key=lambda x: x.getAcc(self.DataName), reverse=True)
        # for i in self.preference:
        #     print(f"{i.getName()} ==> {i.getAcc()}")
        # quit()

    # prepare data to be submitted return % of the historical data of the device
    def Export_data(self, r=1):
        ub = int(r*len(self.preference))
        with open("C:/Users/user/Desktop/localfed/FedMatch/NTDATA.csv", 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            for i in self.preference[:ub]:
                # csv_writer.writerow([i.Region, i.Devt, i.data['mnist'], i.Dicaccuracy['mnist']])
                csv_writer.writerow([i.Producer, i.Region, i.Devt, i.Dicaccuracy['mnist']])

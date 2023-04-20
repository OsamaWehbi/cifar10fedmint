class IoT:
    """Class IoT"""

    def __init__(self, *args):

        # i + 1, "IOT" + str(i + 1), cpu, ram, bandwith,iotTaskTimeNeededForTheJob, data_dct, latencyArray
        
        # id as 1 2...
        self.ID = args[0]
        # id as iot1 iot2...
        self.Name = args[1]
        # generated randomly
        self.cpu = args[2]
        self.ram = args[3]
        self.bandwidth = args[4]
        # time of the job to be done
        self.iottime = args[5]
        # dataname and size of data "minist":300...
        self.data = args[6].copy()
        # generated based on far and close fedserver
        # self.latencydic = args[7].copy()
        # dic latency "Fedname":val
        self.diclatency = args[7].copy()
        # calculated based on cpu,ram,....from each offer
        self.pricedic = args[8].copy()
        # all the initially selected server by the iot
        self.preference = []
        # the value on which array preference is sorted (should be dictionary)
        self.FSweightdic = {}
        # vertical/horizantal...
        # self.datatype = None
        # accuracy first time based on dummy
        self.Dicaccuracy = {h: None for h in self.data.keys()}
        self.DicAvAcc = {h: None for h in self.data.keys()}
        self.Dicstd = {h: None for h in self.data.keys()}
        self.crev = 0
        self.rev = {}
        self.Region = args[9]
        self.Devt = args[10]
        self.Producer = args[11]
        self.part_time = {}
        # self.lstd =[]

    def getID(self):
        return self.ID

    def setID(self, iD):
        self.ID = iD

    def getData(self):
        return self.data

    def setData(self, data):
        self.data = data.copy()

    def getFSweightdic(self):
        return self.FSweightdic

    def setFSweightdic(self, FSweightdic):
        self.FSweightdic = FSweightdic

    def getName(self):
        return self.Name

    def setName(self, name):
        self.Name = name

    def getcrev(self):
        return self.crev

    def setcrev(self, crev):
        self.crev = crev

    def addcrev(self, fedserv):

        i = fedserv
        trev = ((self.getCpu() * float(self.pricedic[i.getName()]["cpu"])) + (self.getRam() * float(self.pricedic[i.getName()]["ram"])) + (
                        self.getBandwidth() * float(self.pricedic[i.getName()]["bandwidth"])) * (1-self.diclatency[i.getName()]))*self.Dicstd[i.DataName]
        self.crev += trev
        self.rev[fedserv.getName()] = trev

    def getBandwidth(self):
        return self.bandwidth

    def setBandwidth(self, bandwidth):
        self.bandwidth = bandwidth

    def getCpu(self):
        return self.cpu

    def setCpu(self, cpu):
        self.cpu = cpu

    def getDicAcc(self):
        return self.Dicaccuracy

    def setDicAcc(self, dicaccuracy):
        self.Dicaccuracy = dicaccuracy

    def getAcc(self, DataName):
        return self.Dicaccuracy[DataName]

    # def setstd(self, nstd, FName):
    #     self.lstd.append(self.Dicstd[FName])
    #     self.Dicstd[FName] = nstd

    def setAcc(self, accuracy, DataName):
        self.Dicaccuracy[DataName] = accuracy
        self.setAvAcc(accuracy,DataName)
        # self.FSweightdic = {}
        # self.preference = []

    def getAvAcc(self):
        return self.DicAvAcc

    def setAvAcc(self, accuracy, DataName):
        if self.DicAvAcc[DataName]:
            self.DicAvAcc[DataName] = (self.DicAvAcc[DataName] + accuracy)/2
        else:
            self.DicAvAcc[DataName] = accuracy

    def getRam(self):
        return self.ram

    def setRam(self, ram):
        self.ram = ram

    def getIoTTaskExecutionTime(self):
        return self.iottime

    def setIoTTime(self, iottime):
        self.iottime = iottime

    def getlatencydic(self):
        return self.diclatency

    def setlatencydic(self, diclatency):
        self.diclatency = diclatency.copy()

    def setpricedic(self, pricedic):
        self.pricedic = pricedic[:]

    def setpreference(self, preference):
        self.preference = preference[:]

    def getpreference(self):
        return self.preference

    def getpricedic(self):
        return self.pricedic

    def getFedlatency(self, Fed):
        return self.diclatency[Fed.getName()]

    def setrev(self, val, fed):
        self.rev[fed] = val

    # def __str__(self):
    #     return f"IoT ID= {self.ID:<5} \tName= {self.Name:<10} \tCPU= {self.cpu} \tRAM= {self.ram} \tBandwidth= {self.bandwidth}"\
    #            f" \tCRev= {round(self.crev,3):<5} Acc= {round(self.Dicaccuracy['kdd'],3):<5} AvAcc= {self.DicAvAcc['kdd']} Std = {self.Dicstd['kdd']}" \
    #            f" Data Available = {self.data} DicLatency = {self.diclatency} Rev: {self.rev}" \
    #            f" \tIottime= {self.iottime :<4} \t"\
    #            f" FSweightdic = {self.FSweightdic} \t" \
    #            f"pricedic = {self.pricedic} \t Array Preference = {self.preference} \t part_time = {self.part_time}"
    def __str__(self):
        return f"IoT ID= {self.ID:<5} Name= {self.Name:<10} CPU= {round(self.cpu,4):<10} RAM= {round(self.ram,4):<10} Bandwidth= {self.bandwidth:<5}"\
               f" Acc= {round(self.Dicaccuracy['cifar10'],3):<5} Std = {round(self.Dicstd['cifar10'],3):<8}"\
               f" Iottime= {round(self.iottime,4) :<10} Data Available = {self.data['cifar10']:<5}"\
               f" Part_time = {self.part_time} \t CRev= {round(self.crev,3):<5} Rev: {self.rev}"
    # def printdev(self):
    #     print(f"IoT ID=  {self.ID} ,Name= {self.Name} , bandwidth= {self.bandwidth} , cpu= {self.cpu}" \
    #           f", ram= {self.ram}, iottime= {self.iottime}, latencydic={self.latencydic}")

    def IoTSW(self):
        # price And latency
        self.FSweightdic = {}
        maxprice = 0
        minlat = 6666
        for i in self.preference:
            # price = (self.getCpu() * float(i.prices["cpu"])) + (self.getRam() * float(i.prices["ram"])) + (
            #             self.getBandwidth() * float(i.prices["bandwidth"])) + (
            #                     float(self.getData()[i.DataName]) * float(i.prices["datasize"]))
            price = (self.getCpu() * float(i.prices["cpu"])) + (self.getRam() * float(i.prices["ram"])) + (
                    self.getBandwidth() * float(i.prices["bandwidth"]))

            # tr = self.prices["cpu"]
            # print(f"{type(price)} val of = {price}")
            # quit()
            if maxprice <= price:
                maxprice = price

            # if minlat >= self.getFedlatency(i):
            #     minlat = self.getFedlatency(i)
            if minlat >= self.diclatency[i.getName()]:
                minlat = self.diclatency[i.getName()]
        for i in self.preference:
            Fprice = (self.getCpu() * float(i.prices["cpu"])) + (self.getRam() * float(i.prices["ram"])) + (
                        self.getBandwidth() * float(i.prices["bandwidth"])) + (
                                float(self.getData()[i.DataName]) * float(i.prices["datasize"]))
            FSweight = ((Fprice / maxprice) * 0.5) + ((minlat / self.diclatency[i.getName()]) * 0.5)

            self.FSweightdic[i.getName()] = FSweight
        # sort the FSweightdic and save the sorting not needed
        self.FSweightdic = dict(sorted(self.FSweightdic.items(), key=lambda item: item[1], reverse=True))
        # sort the preference array based on the weight sorting if the Fsweight is not sorted no pb
        # the sorting will be done automatically but without saving the sorting of the Fsweightdic it just
        # sort the preference
        self.preference = sorted(self.preference, key=lambda x: self.FSweightdic[x.getName()], reverse=True)

    def IoTSort(self):
        self.FSweightdic = {}
        for i in self.preference:
            Fprice = ((self.getCpu() * float(self.pricedic[i.getName()]["cpu"])) + (self.getRam() * float(self.pricedic[i.getName()]["ram"])) + (self.getBandwidth() * float(self.pricedic[i.getName()]["bandwidth"])) * (1-self.diclatency[i.getName()]))*(1-self.Dicstd[i.DataName])
            # Fprice = ((self.getCpu() * float(self.pricedic[i.getName()]["cpu"])) + (self.getRam() * float(self.pricedic[i.getName()]["ram"])) + (self.getBandwidth() * float(self.pricedic[i.getName()]["bandwidth"])) * (1-self.diclatency[i.getName()]))
            # print((self.getBandwidth() * float(i.prices["bandwidth"]))* (1-self.diclatency[i.getName()]))
            # quit()
            self.FSweightdic[i.getName()] = Fprice
        self.FSweightdic = dict(sorted(self.FSweightdic.items(), key=lambda item: item[1], reverse=True))
        # print(self.FSweightdic)
        # quit()
        self.preference = sorted(self.preference, key=lambda x: self.FSweightdic[x.getName()], reverse=True)
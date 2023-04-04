import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Match')
import numpy as np


def federated_matching(IotArr, FedArr):
    IoTs_picks = {}
    Servers_picks = {}
    Servers_NumClients = {}
    for i in IotArr:
        IoTs_picks[i.getName()] = list(dict.keys(i.FSweightdic))
        # print(f"{i.FSweightdic}")
    for i in FedArr:
        # if i.round > 0 and (i.finish == False):
        if (i.finish == False):
            Servers_picks[i.getName()] = list(dict.keys(i.ISweightdic))
            # NumClients[i.getName()] = i.IOTNum
            Servers_NumClients[i.getName()] = i.IOTNum
            # print(i.getName())
            # for y in i.preference:
            #     print(f"{y.getName()} ==> {y.getAcc()}")
    # quit()
    # logger.info('IoT preferences dictionary :')
    # for i, val in IoTs_picks.items():
    #     logger.info('%s : %s', i, val)
    logger.info(IoTs_picks)
    logger.info('Fed preferences dictionary :')
    for i, val in Servers_picks.items():
        logger.info('%s : %s', i, val)
    # print(Servers_picks)
    logger.info('Fed Servers Required IoT\'s :')
    logger.info('%s', Servers_NumClients)
    # print(Servers_NumClients)

    logger.info('Start Matching\'s....')
    # return list of IoT's
    IoTs = list(IoTs_picks.keys())
    # contain iots with their fed after matching
    IoTs_matching = {r: None for r in IoTs_picks.keys()}
    # contain Fed with list of their IOTs
    Servers_matching = {h: [] for h in Servers_picks.keys()}
    try:
        while IoTs:
            # pick iot
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
                if r not in Servers_picks[r_p[0]]:
                    r_p.remove(r_p[0])
                # if r-p iot preference not empty
                if r_p:
                    # pick first preferable Fed
                    h = r_p[0]
                    # let h_p be the preferences of hospital h
                    # get the preference list of the picked fed
                    h_p = Servers_picks[h]
                    # let h_matches be the matched residents for hospital h
                    # get the matched list of iot for the fed (should contain the iot devices after matching done)
                    h_matches = Servers_matching[h]
                    # if there is capacity add
                    if len(h_matches) < Servers_NumClients[h]:
                        IoTs_matching[r] = h
                        Servers_matching[h] += [r]
                    # if not
                    else:

                        # r_rank is a given resident's rank within a hospital's preference list
                        # check the position of the iot in the fed preference
                        r_rank = h_p.index(r)
                        # print(f"rank in fed {r_rank}")
                        # determine the worst rank iot in the matched list from position in the preference list of Fed
                        worst_rank = np.max([h_p.index(i) for i in h_p if i in h_matches])
                        # print(f"worst_rank {worst_rank}")
                        # get the name of the iot device after getting the index
                        # iot device present in the match list of the fed but there is better then him
                        worst_match = h_p[worst_rank]
                        # print(f"worst_match {worst_match}")
                        # if really it is worst then the new one then we should replace it
                        if r_rank < worst_rank:
                            # remove the fed from the match of the resident
                            IoTs_matching[worst_match] = None
                            # remove the worst match from the matching list of the fed that contain iot's
                            h_matches.remove(worst_match)
                            # print(f"first {IoTs_picks[worst_match]} ==> {r}")
                            # remove the Fed from the preference list of the Iot
                            IoTs_picks[worst_match].remove(h)
                            # re add the iot to the Resident list in order to get another change with next fed in
                            # his preference list
                            IoTs.append(worst_match)
                            # add the best iot to the match of the fed
                            IoTs_matching[r] = h
                            # add to the hospital
                            # h_matches += [r]
                            Servers_matching[h] += [r]
                        else:
                            h_p.remove(r)
                            r_p.remove(h)
    except:
        print("out")
        traceback.print_exc()
    for t, val in Servers_matching.items():
        logger.info('%s : %s', t, val)
    Servers_m = {}
    for i, val in Servers_matching.items():
        for j in FedArr:
            if i == j.getName():
                j.Mdic = val

    for i, val in Servers_matching.items():
        for j in val:
            for y in IotArr:
                if j == y.getName():
                    y.part_time[i] +=1
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

    logger.info('Matching Done\'s....')
    # print(IoTs_matching)
    # print('*******************')
    # print(Servers_m)
    # print('*******************')
    # print(Servers_matching)
    # print('*******************')
    return IoTs_matching, Servers_m, Servers_matching, revenuedic

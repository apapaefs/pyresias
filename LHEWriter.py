import numpy as np

## function to initialize lhe file
def init_lhe(filename, sigma, stddev, ECM):
    print('opening Les Houches file', filename, 'for writing.')
    f = open(filename, 'w')
    f.write("<LesHouchesEvents version =\"1.0\">\n")
    f.write("<!--\n")
    f.write("File generated with lhe python writer\n")
    f.write("-->\n")
    f.write("<init>\n")
    f.write("\t11\t -11\t" + str(ECM/2) + "\t" + str(ECM/2) + "\t 0 \t 0 \t 7\t 7 \t 1 \t 1\n")
    f.write("\t" + str(sigma) + "\t" + str(stddev) + "\t1.00000 \t9999\n")
    f.write("</init>\n")
    return f

## function to write a Les Houches event, given some particle information.
def write_lhe(infile, events, shat, debug):

    # example: 
    #status = [-1, -1, 2, 1, 1]
    #momenta = [ pq1, pq2, pz, pem_boosted, pep_boosted ]
    #flavours = [ 1, -1, 23, 13, -13 ]
    #colours = [ 501, 0, 0, 0, 0 ]
    #anticolours = [ 0, 501, 0, 0, 0 ]
    #helicities = [ 1, -1, 0, 1, -1 ]
    #relations = [ [0, 0], [0, 0], [1, 2], [3, 3], [3,3]]
    status = []
    momenta = []
    flavours = []
    colours = []
    anticolours = []
    helicities = []
    relations = []
    # loop over events and fill in the lists:
    for eventno, event in enumerate(events):
        # count the number of gluons
        ng = 0
        status = []
        momenta = []
        flavours = []
        colours = []
        anticolours = []
        helicities = []
        relations = []
        for p in event:
            momenta.append([p[2],p[3],p[4],p[5]])
            status.append(p[1])
            if p[1] == -1:
                relations.append([0,0])
            elif p[1] == 1:
                relations.append([1,2])
            flavours.append(p[0])
            helicities.append(1)
            if abs(p[0]) == 11: # electrons
                colours.append(0)
                anticolours.append(0)
            if abs(p[0])>0 and abs(p[0])<6: # q or qbar -> this only works for the specific process e+e- -> qqbar! 
                if p[0] < 0:
                    colours.append(0)
                    anticolours.append(501)
                elif p[0] > 0:
                    colours.append(501)
                    anticolours.append(0)
            if p[0] == 21: # WARNING: gluons are singlets for now!
                ng += 1
                colours.append(500 + 2 * ng)
                anticolours.append(500 + 2 * ng)
        if debug:
            print('writing event')
            print(shat)
            print(str(np.sqrt(shat)))
        infile.write("<event>\n")
        infile.write(str(len(momenta)) + "\t 9999\t 1.000000\t " + str(np.sqrt(shat)) + "\t 0.0078125 \t 0.1187\n")
        for i in range(0,len(momenta)):
            p = momenta[i]            
            mass = 0 # WARNING: everything is massless for now!
            particlestring = str(flavours[i]) + "\t" + str(status[i]) + "\t" + str(relations[i][0]) + "\t" + str(relations[i][1]) + "\t" +  str(colours[i]) + "\t" + str(anticolours[i]) + "\t" + str(p[0]) + "\t" + str(p[1]) + "\t" + str(p[2]) + "\t" + str(p[3]) + "\t" + str(mass) + "\t0\t" + str(helicities[i]) + "\n"
            infile.write(particlestring)
            if debug:
                print(particlestring)
        infile.write("</event>\n")

## function to finalize the lhe file
def finalize_lhe(infile):
    print('closing Les Houches file')
    infile.write("</LesHouchesEvents>\n")
    infile.close()

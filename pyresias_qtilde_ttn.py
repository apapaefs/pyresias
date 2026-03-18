from random import *  # random numbers
import gzip
import json
import math
import sys
from functools import lru_cache

import numpy as np
import scipy
from optparse import OptionParser
from scipy import optimize

from alphaS_HW import *

try:
    from prettytable import PrettyTable
except ImportError:
    class PrettyTable:  # minimal fallback for environments without prettytable
        def __init__(self, field_names):
            self.field_names = field_names
            self.rows = []

        def add_row(self, row):
            self.rows.append(row)

        def __str__(self):
            widths = [len(str(name)) for name in self.field_names]
            for row in self.rows:
                for idx, value in enumerate(row):
                    widths[idx] = max(widths[idx], len(str(value)))
            header = " | ".join(str(name).ljust(widths[idx]) for idx, name in enumerate(self.field_names))
            sep = "-+-".join("-" * width for width in widths)
            body = [" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) for row in self.rows]
            return "\n".join([header, sep] + body)

        __repr__ = __str__

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

################################################
print("\nPyresias-qTilde-TTN: a toy parton shower\n")
# a direct derivative of pyresias_qtilde.py
# with fixed-history TTN diagnostics added on top
#################################################


def PrintEmissions(EmissionsArray):
    tbl = PrettyTable(["#", "Evo scale [GeV]", "1-z", "pT [GeV]", "virt. mass in a->bc [GeV]"])
    for i in range(len(EmissionsArray)):
        tbl.add_row([i, EmissionsArray[i][0], 1 - EmissionsArray[i][1], EmissionsArray[i][2], EmissionsArray[i][3]])
    print(tbl)


def PrintMomenta(MomentaArray):
    tbl = PrettyTable(["n", "id", "status", "px [GeV]", "py [GeV]", "pz [GeV]", "E [GeV]", "m [GeV]"])
    for i in range(len(MomentaArray)):
        tbl.add_row(
            [
                i,
                MomentaArray[i][0],
                MomentaArray[i][1],
                MomentaArray[i][2],
                MomentaArray[i][3],
                MomentaArray[i][4],
                MomentaArray[i][5],
                MomentaArray[i][6],
            ]
        )
    print(tbl)


#############
# SWITCHES: #
#############

debug = False
printevents = False
inputfile = ""
outputfile = ""
Nshower = int(1e99)
Qc = 0.935
pTmin = 0.900
ttn_max_gluons = 6
ttn_truncation_chis = (3, 8, 16, 32)
skip_output = False
ttn_reportfile = ""


##########################
# LIGHTWEIGHT LHE I/O    #
##########################


def read_momenta(inputline):
    pid = int(inputline.split()[0])
    status = int(inputline.split()[1])
    col = int(inputline.split()[4])
    acol = int(inputline.split()[5])
    px = float(inputline.split()[6])
    py = float(inputline.split()[7])
    pz = float(inputline.split()[8])
    e = float(inputline.split()[9])
    m = float(inputline.split()[10])
    return [pid, status, px, py, pz, e, m, col, acol]


def readlhefile(infile):
    if infile.endswith(".gz"):
        my_open = gzip.open
    else:
        my_open = open
    infile_read = my_open(infile, "rt")
    reading_event = False
    events = []
    weights = []
    multiweights = []
    weight = 1.0
    for line in infile_read:
        if "<event>" in line:
            particles = []
            multiweight = {}
            reading_event = True
        if reading_event is True:
            if "</event>" in line:
                reading_event = False
                events.append(particles)
                weights.append(weight)
                multiweights.append(multiweight)
            if len(line.split()) == 6:
                weight = float(line.split()[2])
            if len(line.split()) == 13:
                particles.append(read_momenta(line))
            if len(line.split()) == 4 and line.split()[1].startswith("id="):
                key = line.split()[1].replace("id=", "").replace(">", "").replace("'", "")
                multiweight[key] = float(line.split()[2])
    infile_read.close()
    return events, weights, multiweights


def init_lhe(filename, sigma, stddev, ECM):
    print("opening Les Houches file", filename, "for writing.")
    fout = open(filename, "w")
    fout.write("<LesHouchesEvents version =\"1.0\">\n")
    fout.write("<!--\n")
    fout.write("File generated with lhe python writer\n")
    fout.write("-->\n")
    fout.write("<init>\n")
    fout.write("\t11\t -11\t" + str(ECM / 2) + "\t" + str(ECM / 2) + "\t 0 \t 0 \t 7\t 7 \t 1 \t 1\n")
    fout.write("\t" + str(sigma) + "\t" + str(stddev) + "\t1.00000 \t9999\n")
    fout.write("</init>\n")
    return fout


def write_lhe(infile, events, shat, debug):
    for event in events:
        ng = 0
        status = []
        momenta = []
        flavours = []
        colours = []
        anticolours = []
        helicities = []
        relations = []
        for p in event:
            momenta.append([p[2], p[3], p[4], p[5]])
            status.append(p[1])
            if p[1] == -1:
                relations.append([0, 0])
            elif p[1] == 1:
                relations.append([1, 2])
            flavours.append(p[0])
            helicities.append(1)
            if abs(p[0]) == 11:
                colours.append(0)
                anticolours.append(0)
            if abs(p[0]) > 0 and abs(p[0]) < 6:
                if p[0] < 0:
                    colours.append(0)
                    anticolours.append(501)
                elif p[0] > 0:
                    colours.append(501)
                    anticolours.append(0)
            if p[0] == 21:
                ng += 1
                colours.append(500 + 2 * ng)
                anticolours.append(500 + 2 * ng)
        infile.write("<event>\n")
        infile.write(str(len(momenta)) + "\t 9999\t 1.000000\t " + str(np.sqrt(shat)) + "\t 0.0078125 \t 0.1187\n")
        for i in range(0, len(momenta)):
            p = momenta[i]
            mass = 0
            particlestring = (
                str(flavours[i])
                + "\t"
                + str(status[i])
                + "\t"
                + str(relations[i][0])
                + "\t"
                + str(relations[i][1])
                + "\t"
                + str(colours[i])
                + "\t"
                + str(anticolours[i])
                + "\t"
                + str(p[0])
                + "\t"
                + str(p[1])
                + "\t"
                + str(p[2])
                + "\t"
                + str(p[3])
                + "\t"
                + str(mass)
                + "\t0\t"
                + str(helicities[i])
                + "\n"
            )
            infile.write(particlestring)
            if debug:
                print(particlestring)
        infile.write("</event>\n")


def finalize_lhe(infile):
    print("closing Les Houches file")
    infile.write("</LesHouchesEvents>\n")
    infile.close()


##########################
# COMMAND LINE ARGUMENTS #
##########################


def build_parser():
    parser = OptionParser(usage="%prog [options] [inputfile]", version="Pyresias 0.2")
    parser.add_option("-d", "--debug", dest="debug", default=False, action="store_true", help="Print debugging to screen")
    parser.add_option(
        "-p", "--printevents", dest="printevents", default=False, action="store_true", help="Print showered events to screen"
    )
    parser.add_option("-n", "--nshower", dest="nshower", default=Nshower, help="Set the number of events to shower")
    parser.add_option("-c", dest="Qc", default=Qc, help="Set the cutoff scale for the evolution")
    parser.add_option("-o", dest="output", default=outputfile, help="Set the output file name")
    parser.add_option("--ptmin", dest="pTmin", default=pTmin, help="Set the minimum pT in the shower")
    parser.add_option(
        "--ttn-max-gluons",
        dest="ttn_max_gluons",
        default=ttn_max_gluons,
        help="Maximum total emitted gluons for exact dense TTN diagnostics",
    )
    parser.add_option(
        "--skip-output",
        dest="skip_output",
        default=False,
        action="store_true",
        help="Run the shower and TTN diagnostics without writing an output LHE file",
    )
    parser.add_option(
        "--ttn-trunc-chis",
        dest="ttn_truncation_chis",
        default=",".join(str(value) for value in ttn_truncation_chis),
        help="Comma-separated bond dimensions used for frontier truncation checks",
    )
    parser.add_option(
        "--ttn-report",
        dest="ttn_reportfile",
        default=ttn_reportfile,
        help="Write TTN diagnostics to the given JSON report file",
    )
    return parser


def configure_from_args(argv=None):
    global debug, printevents, Nshower, Qc, outputfile, inputfile, pTmin, ttn_max_gluons, ttn_truncation_chis, skip_output, ttn_reportfile

    parser = build_parser()
    options, args = parser.parse_args(args=argv)

    debug = options.debug
    printevents = options.printevents
    Nshower = int(options.nshower)
    Qc = float(options.Qc)
    outputfile = str(options.output)
    pTmin = float(options.pTmin)
    ttn_max_gluons = int(options.ttn_max_gluons)
    ttn_truncation_chis = tuple(int(value.strip()) for value in str(options.ttn_truncation_chis).split(",") if value.strip() != "")
    skip_output = options.skip_output
    ttn_reportfile = str(options.ttn_reportfile)

    if len(args) < 1:
        parser.error("An input file is required!")

    inputfile = args[0]
    if outputfile == "":
        outputfile = inputfile.replace(".lhe", "").replace(".gz", "") + "_ttn.hepmc"
    if ttn_reportfile == "":
        ttn_reportfile = outputfile.replace(".hepmc", "_report.json")


#################################################

# initialize alphaS class: pass the value of alphaS at mz, and mz
aS = alphaS(0.1074, 91.1876)

# CMW scheme:
CMW = "None"  # 'Linear' or 'Factor' or 'None'


def Kg():
    Nf = 5
    return 3.0 * (67.0 / 18.0 - 1.0 / 6.0 * np.pi**2) - 5.0 / 9.0 * Nf


def Pqq(z, t, Qcut, aSover):
    if CMW == "Linear" or CMW == "Factor":
        aS_local = alphaS(t, z, Qcut, aSover)
        return CF * (1 + aS_local / 2 / np.pi * Kg() + z**2) / (1.0 - z)
    elif CMW == "None":
        return CF * (1.0 + z**2) / (1.0 - z)


def Pqq_over(z):
    return 2.0 * CF / (1.0 - z)


def scale_of_alphaS(t, z):
    return z * (1 - z) * np.sqrt(t)


def alphaS(t, z, Qcut, aSover):
    scale = scale_of_alphaS(t, z)
    if scale < Qcut and CMW == "None" or CMW == "Linear":
        scale = Qcut
    if CMW == "Linear":
        CMWFactor = 1 + Kg() * aS.alphasQ(scale) / 2.0 / np.pi
        return aS.alphasQ(scale) / 2.0 / np.pi * CMWFactor
    elif CMW == "Factor":
        Nf = 5
        CMWFactor = np.exp(-(67 - 3 * np.pi**2 - 10 / 3 * Nf) / (33 - 2 * Nf))
        scale *= CMWFactor
        if scale < Qcut:
            scale = Qcut
        return aS.alphasQ(scale) / 2.0 / np.pi
    elif CMW == "None":
        return aS.alphasQ(scale) / 2.0 / np.pi


def tGamma(z, aSover):
    return -2.0 * aSover * CF * np.log1p(-z)


def inversetGamma(r, aSover):
    return 1.0 - np.exp(-0.5 * r / CF / aSover)


def zp_over(t, cut):
    return 1.0 - np.sqrt(cut**2 / t)


def zm_over(t, cut):
    return np.sqrt(cut**2 / t)


def get_alphaS_over(Qcut):
    minscale = Qcut
    if minscale < Qcut:
        scale = minscale
    else:
        scale = Qcut
    if CMW == "Linear":
        CMWFactor = 1 + Kg() * aS.alphasQ(scale) / 2.0 / np.pi
        alphaS_over = aS.alphasQ(scale) / 2.0 / np.pi * CMWFactor
    elif CMW == "Factor":
        Nf = 5
        CMWFactor = np.exp(-(67 - 3 * np.pi**2 - 10 / 3 * Nf) / (33 - 2 * Nf))
        scale *= CMWFactor
        alphaS_over = aS.alphasQ(scale) / 2.0 / np.pi
    elif CMW == "None":
        alphaS_over = aS.alphasQ(scale) / 2.0 / np.pi
    if debug:
        print("alpha_S overestimate set to", alphaS_over, "for scale=", scale, "GeV")
    return alphaS_over


def Get_zEmission(t, Qcut, R, aSover):
    return inversetGamma(
        tGamma(zm_over(t, Qcut), aSover) + R * (tGamma(zp_over(t, Qcut), aSover) - tGamma(zm_over(t, Qcut), aSover)),
        aSover,
    )


def Get_pTsq(t, z):
    return z**2 * (1 - z) ** 2 * t


def Get_mvirtsq(t, z):
    return z * (1 - z) * t


def Get_tEmission_direct(Q, Qcut, R, aSover):
    upper = tGamma(zp_over(Q**2, Qcut), aSover)
    lower = tGamma(zm_over(Q**2, Qcut), aSover)
    if lower > upper:
        if debug:
            print("\tEmission fails due upper < lower")
        return Q**2, [], False
    c = 1 / (upper - lower)
    tEm_sol = Q**2 * R**c
    if math.isnan(tEm_sol) or tEm_sol < 4 * Qcut**2:
        if debug:
            print("\tEmission fails due to NaN tEm or tEm_sol < 4*Qcut**2, tEm_sol=", tEm_sol)
        return Q**2, [], False
    return tEm_sol, [], True


def Generate_Emission(Q, Qcut, aSover):
    generated = True
    R1 = random()
    R2 = random()
    R3 = random()
    R4 = random()
    tEm, results, continueEvolution = Get_tEmission_direct(Q, Qcut, R1, aSover)
    if continueEvolution == False:
        zEm = 1.0
        pTsqEm = 0.0
        MsqEm = 0.0
        if debug:
            print("continueEvolution is False")
        return tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution
    if debug:
        print("\tcandidate emission scale, sqrt(tEm)=", np.sqrt(tEm))
    if tEm < 4 * Qcut**2:
        generated = False
    zp_true = zp_over(tEm, Qcut)
    zm_true = zm_over(tEm, Qcut)
    if zm_true < 0 or zp_true < 0:
        generated = False
    if zm_true > zp_true:
        generated = False
    zEm = Get_zEmission(Q**2, Qcut, R2, aSover)
    if debug:
        print("\t\tcandidate momentum fraction, zEm=", zEm)
    if zEm < zm_true or zEm > zp_true:
        generated = False
    pTsqEm = Get_pTsq(tEm, zEm)
    if debug:
        print("\t\tcandidate transverse momentum =", np.sqrt(pTsqEm))
    if pTsqEm < pTmin**2:
        generated = False
    if pTsqEm < 0.0:
        generated = False
    if Pqq(zEm, tEm, Qcut, aSover) / Pqq_over(zEm) < R3:
        generated = False
    if alphaS(tEm, zEm, Qcut, aSover) / aSover < R4:
        generated = False
    MsqEm = Get_mvirtsq(tEm, zEm)
    if debug and generated == True:
        print("\t\t---> Emission accepted!")
    if generated == False:
        zEm = 1.0
        pTsqEm = 0.0
        MsqEm = 0.0
    return tEm, zEm, pTsqEm, MsqEm, generated, continueEvolution


ACTIVE_FLAVOURS = [1, 2, 3, 4, 5]


def PrimitiveOverestimate(branch_type, z):
    if branch_type == "q->qg":
        return -2.0 * CF * np.log1p(-z)
    if branch_type == "g->gg":
        return CA * (np.log(z) - np.log1p(-z))
    if branch_type == "g->qqbar":
        return len(ACTIVE_FLAVOURS) * TR * z
    raise ValueError("Unknown branch type " + str(branch_type))


def InversePrimitiveOverestimate(branch_type, value):
    if branch_type == "q->qg":
        return 1.0 - np.exp(-0.5 * value / CF)
    if branch_type == "g->gg":
        return 1.0 / (1.0 + np.exp(-value / CA))
    if branch_type == "g->qqbar":
        return value / (len(ACTIVE_FLAVOURS) * TR)
    raise ValueError("Unknown branch type " + str(branch_type))


def SplittingKernel(branch_type, z, t, Qcut, aSover):
    if branch_type == "q->qg":
        return Pqq(z, t, Qcut, aSover)
    if branch_type == "g->gg":
        return CA * ((1.0 - z) / z + z / (1.0 - z) + z * (1.0 - z))
    if branch_type == "g->qqbar":
        return len(ACTIVE_FLAVOURS) * TR * (1.0 - 2.0 * z * (1.0 - z))
    raise ValueError("Unknown branch type " + str(branch_type))


def SplittingOverestimate(branch_type, z):
    if branch_type == "q->qg":
        return Pqq_over(z)
    if branch_type == "g->gg":
        return CA * (1.0 / z + 1.0 / (1.0 - z))
    if branch_type == "g->qqbar":
        return len(ACTIVE_FLAVOURS) * TR
    raise ValueError("Unknown branch type " + str(branch_type))


def AllowedBranchings(pid):
    if abs(pid) > 0 and abs(pid) < 6:
        return ["q->qg"]
    if pid == 21:
        return ["g->gg", "g->qqbar"]
    return []


def Generate_Channel_Candidate(branch_type, Q, Qcut, aSover):
    R1 = random()
    R2 = random()
    zmin_over = zm_over(Q**2, Qcut)
    zmax_over = zp_over(Q**2, Qcut)
    if zmin_over >= zmax_over or zmin_over <= 0.0 or zmax_over >= 1.0:
        return {"continueEvolution": False, "branch_type": branch_type}
    integral = PrimitiveOverestimate(branch_type, zmax_over) - PrimitiveOverestimate(branch_type, zmin_over)
    if not np.isfinite(integral) or integral <= 0.0:
        return {"continueEvolution": False, "branch_type": branch_type}
    rate = aSover * integral
    if rate <= 0.0:
        return {"continueEvolution": False, "branch_type": branch_type}
    tEm = Q**2 * R1 ** (1.0 / rate)
    if math.isnan(tEm) or tEm < 4.0 * Qcut**2:
        return {"continueEvolution": False, "branch_type": branch_type}
    primitive = PrimitiveOverestimate(branch_type, zmin_over) + R2 * integral
    zEm = InversePrimitiveOverestimate(branch_type, primitive)
    return {"continueEvolution": True, "branch_type": branch_type, "tEm": tEm, "zEm": zEm}


def Generate_Branching(pid, Q, Qcut, aSover):
    currentQ = Q
    fac_cutoff = 4.0
    while currentQ > np.sqrt(fac_cutoff) * Qcut:
        candidates = []
        for branch_type in AllowedBranchings(pid):
            candidate = Generate_Channel_Candidate(branch_type, currentQ, Qcut, aSover)
            if candidate["continueEvolution"]:
                candidates.append(candidate)
        if len(candidates) == 0:
            return None

        candidate = max(candidates, key=lambda item: item["tEm"])
        tEm = candidate["tEm"]
        zEm = candidate["zEm"]
        branch_type = candidate["branch_type"]

        if tEm < fac_cutoff * Qcut**2:
            return None

        zm_true = zm_over(tEm, Qcut)
        zp_true = zp_over(tEm, Qcut)
        generated = True
        if zEm < zm_true or zEm > zp_true or zm_true > zp_true:
            generated = False

        pTsqEm = Get_pTsq(tEm, zEm)
        if pTsqEm < pTmin**2 or pTsqEm < 0.0:
            generated = False

        if SplittingKernel(branch_type, zEm, tEm, Qcut, aSover) / SplittingOverestimate(branch_type, zEm) < random():
            generated = False
        if alphaS(tEm, zEm, Qcut, aSover) / aSover < random():
            generated = False

        MsqEm = Get_mvirtsq(tEm, zEm)
        if generated:
            phi = (2 * random() - 1) * np.pi
            qtilde = np.sqrt(tEm)
            if branch_type == "q->qg":
                if pid > 0:
                    child_pids = [pid, 21]
                else:
                    child_pids = [pid, 21]
            elif branch_type == "g->gg":
                child_pids = [21, 21]
            elif branch_type == "g->qqbar":
                flavour = choice(ACTIVE_FLAVOURS)
                child_pids = [flavour, -flavour]
            else:
                child_pids = []
            return {
                "type": branch_type,
                "t": tEm,
                "qtilde": qtilde,
                "z": zEm,
                "pT": np.sqrt(pTsqEm),
                "msq": MsqEm,
                "phi": phi,
                "child_pids": child_pids,
            }
        currentQ = np.sqrt(tEm)
    return None


def CountTreeBranchings(node):
    counts = {"q->qg": 0, "g->gg": 0, "g->qqbar": 0}
    if node["branch"] is not None:
        counts[node["branch"]["type"]] += 1
        for child in node["children"]:
            child_counts = CountTreeBranchings(child)
            for key in counts:
                counts[key] += child_counts[key]
    return counts


def TreeContainsGluonBranching(node):
    if node["branch"] is None:
        return False
    if node["branch"]["type"] != "q->qg":
        return True
    return any(TreeContainsGluonBranching(child) for child in node["children"])


def FlattenLinearEmissions(node):
    emissions = []
    current = node
    while current["branch"] is not None and current["branch"]["type"] == "q->qg":
        branch = current["branch"]
        emissions.append([branch["t"], branch["z"], branch["pT"], branch["msq"], branch["phi"]])
        current = current["children"][0]
    return emissions


def EvolveParticle(pid, Qstart, Qmin, aSover):
    node = {"pid": pid, "start_scale": Qstart, "branch": None, "children": []}
    if Qstart <= 0.0:
        return node
    branching = Generate_Branching(pid, Qstart, Qmin, aSover)
    if branching is None:
        return node
    node["branch"] = branching
    left_pid = branching["child_pids"][0]
    right_pid = branching["child_pids"][1]
    node["children"] = [
        EvolveParticle(left_pid, branching["z"] * branching["qtilde"], Qmin, aSover),
        EvolveParticle(right_pid, (1.0 - branching["z"]) * branching["qtilde"], Qmin, aSover),
    ]
    return node


def CollectLeafMomenta(node, alpha_parent, qT_parent, pdotn, p, n, Momenta):
    if node["branch"] is None:
        qT2 = qT_parent[0] ** 2 + qT_parent[1] ** 2
        beta = qT2 / (2.0 * alpha_parent * pdotn)
        px = alpha_parent * p[2] + beta * n[2] + qT_parent[0]
        py = alpha_parent * p[3] + beta * n[3] + qT_parent[1]
        pz = alpha_parent * p[4] + beta * n[4]
        E = alpha_parent * p[5] + beta * n[5]
        Momenta.append([node["pid"], 1, px, py, pz, E, 0])
        return

    branch = node["branch"]
    z = branch["z"]
    pT = branch["pT"]
    phi = branch["phi"]
    kT = np.array([pT * np.cos(phi), pT * np.sin(phi)])
    qT_left = z * qT_parent + kT
    qT_right = (1.0 - z) * qT_parent - kT
    CollectLeafMomenta(node["children"][0], z * alpha_parent, qT_left, pdotn, p, n, Momenta)
    CollectLeafMomenta(node["children"][1], (1.0 - z) * alpha_parent, qT_right, pdotn, p, n, Momenta)


def Shower(particles, Qmin, aSover):
    AllMomenta = []
    JetMomenta = []
    JetHistories = []
    for p in particles:
        if abs(p[0]) == 11:
            AllMomenta.append(p)
        elif abs(p[0]) > 0 and abs(p[0]) < 6 and p[1] == 1:
            if debug:
                print("Showering quark:", p[0])
            ppartner, Q2start = find_color_partner(p, particles)
            tree = EvolveParticle(p[0], np.sqrt(max(Q2start, 0.0)), Qmin, aSover)
            Momenta = reconstructSudakov(p, ppartner, tree)
            RotatedMomenta = RotateMomentaLab(p, Momenta)
            for Mom in RotatedMomenta:
                AllMomenta.append(Mom)
            JetMomenta.append([p, RotatedMomenta])
            JetHistories.append(
                {
                    "branch_type": "q" if p[0] > 0 else "qbar",
                    "progenitor": p,
                    "partner": ppartner,
                    "Q2start": Q2start,
                    "tree": tree,
                    "emissions": FlattenLinearEmissions(tree),
                    "branch_counts": CountTreeBranchings(tree),
                    "has_gluon_branching": TreeContainsGluonBranching(tree),
                    "momenta": RotatedMomenta,
                }
            )
    return AllMomenta, JetMomenta, JetHistories


def find_color_partner(part, particles):
    partner = []
    for pc in particles:
        if part[7] == pc[8] and part[8] == pc[7]:
            partner = pc
            if debug:
                print("Color partner of particle", part, "found:", pc)
    if len(partner) != 0:
        Q2start = (part[5] + partner[5]) ** 2 - (
            (part[2] + partner[2]) ** 2 + (part[3] + partner[3]) ** 2 + (part[4] + partner[4]) ** 2
        )
    else:
        Q2start = 0
    return partner, Q2start


def reconstructSudakov(pin, nin, tree):
    Momenta = []
    pmag = np.sqrt(pin[2] ** 2 + pin[3] ** 2 + pin[4] ** 2)
    p = [pin[0], pin[1], 0, 0, pmag, pmag, 0]
    nmag = np.sqrt(nin[2] ** 2 + nin[3] ** 2 + nin[4] ** 2)
    n = [nin[0], nin[1], 0, 0, -nmag, nmag, 0]
    pdotn = dot4vec(p, n)
    CollectLeafMomenta(tree, 1.0, np.array([0.0, 0.0]), pdotn, p, n, Momenta)
    return Momenta


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def GetRotationMatrixAB(a, b):
    acrossb = np.cross(a, b)
    if np.linalg.norm(acrossb) > 1e-12:
        x = unit_vector(acrossb)
        theta = angle_between(a, b)
        I = np.identity(3)
        A = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        R = I + A * np.sin(theta) + np.dot(A, A) * (1 - np.cos(theta))
    else:
        return np.identity(3)
    return R


def RotateMomentaLab(p, Momenta):
    RotatedMomenta = []
    pmag = np.sqrt(p[2] ** 2 + p[3] ** 2 + p[4] ** 2)
    pzonly = np.array([0, 0, pmag])
    Rmatrix = GetRotationMatrixAB(pzonly, np.array([p[2], p[3], p[4]]))
    for pm in Momenta:
        a = np.array([pm[2], pm[3], pm[4]])
        c = np.dot(Rmatrix, a)
        RotatedMomenta.append([pm[0], 1, c[0], c[1], c[2], pm[5], pm[6]])
    return RotatedMomenta


def CheckMomentumConservation(Momenta):
    totalmom = np.array([0, 0, 0])
    for pm in Momenta:
        if pm[1] == 1:
            totalmom = totalmom + np.array([pm[2], pm[3], pm[4]])
    return totalmom


def GlobalMomCons(showeredParticles, showeredJets):
    sqrthatS = 0
    pj2array = []
    qj2array = []
    newqarray = []
    oldparray = []
    Rqparray = []
    for jet in showeredJets:
        oldp = np.array([jet[0][2], jet[0][3], jet[0][4], jet[0][5]])
        pj2 = jet[0][2] ** 2 + jet[0][3] ** 2 + jet[0][4] ** 2
        sqrthatS += jet[0][5]
        qj = np.array([0, 0, 0, 0])
        for p in jet[1]:
            qj = qj + np.array([p[2], p[3], p[4], p[5]])
            qj2 = qj[3] ** 2 - qj[0] ** 2 - qj[1] ** 2 - qj[2] ** 2
        if math.isnan(qj2):
            qj2 = 0
        Rqp = GetRotationMatrixAB(np.array([qj[0], qj[1], qj[2]]), np.array([oldp[0], oldp[1], oldp[2]]))
        pj2array.append(pj2)
        qj2array.append(qj2)
        newqarray.append(qj)
        oldparray.append(oldp)
        Rqparray.append(Rqp)

    def keqn(x):
        kres = 0
        for i in range(len(pj2array)):
            kres += np.sqrt(x * pj2array[i] + qj2array[i])
        kres = kres - sqrthatS
        return kres

    kres = np.sqrt(optimize.root(keqn, 0.99).x[0])
    if debug:
        print("kres=", kres)

    showeredParticlesBoosted = []
    for p in showeredParticles:
        if abs(p[0]) == 11:
            showeredParticlesBoosted.append(p)
    showeredJetsBoosted = []

    either_radiated = any(len(jet[1]) > 1 for jet in showeredJets)
    if either_radiated is False:
        for jj, jet in enumerate(showeredJets):
            showeredJetsBoosted.append(jet[1][0])
            showeredParticlesBoosted.append(jet[1][0])
    else:
        for jj, jet in enumerate(showeredJets):
            showeredJetBoosted = []
            boostvec = getBoostBeta(kres, newqarray[jj], oldparray[jj])
            for p in jet[1]:
                protated = rotate(p, Rqparray[jj])
                pboosted = boost(np.array([protated[2], protated[3], protated[4], protated[5]]), boostvec)
                showeredParticlesBoosted.append([p[0], p[1], pboosted[0], pboosted[1], pboosted[2], pboosted[3], p[6]])
                showeredJetBoosted.append([p[0], p[1], pboosted[0], pboosted[1], pboosted[2], pboosted[3], p[6]])
            showeredJetsBoosted.append(showeredJetBoosted)
    return showeredParticlesBoosted


def getBoostBeta(k, newq, oldp):
    qs = newq[0] ** 2 + newq[1] ** 2 + newq[2] ** 2
    q = np.sqrt(qs)
    Q2 = newq[3] ** 2 - newq[0] ** 2 - newq[1] ** 2 - newq[2] ** 2
    kp = k * np.sqrt(oldp[0] ** 2 + oldp[1] ** 2 + oldp[2] ** 2)
    kps = kp**2
    betam = (q * newq[3] - kp * np.sqrt(kps + Q2)) / (kps + qs + Q2)
    beta = betam * (k / kp) * np.array([oldp[0], oldp[1], oldp[2]])
    if betam >= 0:
        return beta
    else:
        return np.array([0, 0, 0])


def boost(fourvector, betavec):
    if betavec.all() == 0:
        return fourvector
    betax = betavec[0]
    betay = betavec[1]
    betaz = betavec[2]
    beta = np.sqrt(betavec[0] ** 2 + betavec[1] ** 2 + betavec[2] ** 2)
    boosted = [0, 0, 0, 0]
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    boosted[3] = gamma * (fourvector[3] - betax * fourvector[0] - betay * fourvector[1] - betaz * fourvector[2])
    boosted[0] = (
        -gamma * betax * fourvector[3]
        + (1 + (gamma - 1) * betax**2 / beta**2) * fourvector[0]
        + (gamma - 1) * betax * betay * fourvector[1] / beta**2
        + (gamma - 1) * betax * betaz / beta**2 * fourvector[2]
    )
    boosted[1] = (
        -gamma * betay * fourvector[3]
        + (gamma - 1) * betay * betax * fourvector[0] / beta**2
        + (1 + (gamma - 1) * betay**2 / beta**2) * fourvector[1]
        + (gamma - 1) * betay * betaz * fourvector[2] / beta**2
    )
    boosted[2] = (
        -gamma * betaz * fourvector[3]
        + (gamma - 1) * betaz * betax * fourvector[0] / beta**2
        + (gamma - 1) * betaz * betay * fourvector[1] / beta**2
        + (1 + (gamma - 1) * betaz**2 / beta**2) * fourvector[2]
    )
    return boosted


def rotate(p, Rmatrix):
    a = np.array([p[2], p[3], p[4]])
    c = np.dot(Rmatrix, a)
    RotatedMomentum = np.array([p[0], 1, c[0], c[1], c[2], p[5], p[6]])
    return RotatedMomentum


def dot4vec(p, n):
    return p[5] * n[5] - p[2] * n[2] - p[3] * n[3] - p[4] * n[4]


############################
# FIXED-HISTORY TTN LAYER  #
############################


def fundamental_generators():
    gens = np.zeros((8, 3, 3), dtype=np.complex128)
    gens[0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    gens[1] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    gens[2] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    gens[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
    gens[4] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    gens[5] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    gens[6] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    gens[7] = (1.0 / np.sqrt(3.0)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128)
    return 0.5 * gens


TTN_TF = fundamental_generators()
TTN_TBAR = -np.conjugate(TTN_TF)
TTN_HARD_SINGLET = np.eye(3, dtype=np.complex128) / np.sqrt(3.0)
TTN_EPS = 1e-12


def color_dimension(pid):
    if pid == 21:
        return 8
    return 3


def structure_constants():
    fabc = np.zeros((8, 8, 8), dtype=np.float64)
    for a in range(8):
        for b in range(8):
            comm = TTN_TF[a] @ TTN_TF[b] - TTN_TF[b] @ TTN_TF[a]
            for c in range(8):
                value = -2j * np.trace(comm @ TTN_TF[c])
                fabc[a, b, c] = float(np.real_if_close(value))
    return fabc


TTN_FABC = structure_constants()
# Indices are (emitted gluon, continuing child, parent).
TTN_TG = -1j * TTN_FABC


def DenseDimensionCap(max_dense_gluons):
    return int(9 * (8**max_dense_gluons))


def DenseStateDimensionFromLeaves(leaves):
    dimension = 1
    for leaf in leaves:
        dimension *= int(leaf["dim"])
    return int(dimension)


def BranchColorTensor(parent_pid, branch):
    branch_type = branch["type"]
    if branch_type == "q->qg":
        emission_tensor = TTN_TF if parent_pid > 0 else TTN_TBAR
        return np.transpose(emission_tensor, (2, 1, 0))
    if branch_type == "g->gg":
        return np.transpose(TTN_TG, (2, 1, 0))
    if branch_type == "g->qqbar":
        return TTN_TF.copy()
    raise ValueError("Unknown branch type " + str(branch_type))


def CollectTreeLeaves(node, root_label, path=()):
    if node["branch"] is None:
        return [
            {
                "root": root_label,
                "path": tuple(path),
                "pid": int(node["pid"]),
                "dim": int(color_dimension(node["pid"])),
            }
        ]
    leaves = []
    for child_idx, child in enumerate(node["children"]):
        leaves.extend(CollectTreeLeaves(child, root_label, path + (child_idx,)))
    return leaves


def BuildTreeColorMap(node):
    if node["branch"] is None:
        dim = color_dimension(node["pid"])
        return np.eye(dim, dtype=np.complex128)

    child0_map = BuildTreeColorMap(node["children"][0])
    child1_map = BuildTreeColorMap(node["children"][1])
    local_tensor = BranchColorTensor(node["pid"], node["branch"])

    combined = np.tensordot(local_tensor, child0_map, axes=([1], [0]))
    perm = [0] + list(range(2, combined.ndim)) + [1]
    combined = np.transpose(combined, perm)
    combined = np.tensordot(combined, child1_map, axes=([-1], [0]))
    return combined


def BuildEventColorState(q_tree, qbar_tree):
    q_map = BuildTreeColorMap(q_tree)
    qbar_map = BuildTreeColorMap(qbar_tree)
    combined = np.tensordot(TTN_HARD_SINGLET, q_map, axes=([0], [0]))
    combined = np.tensordot(combined, qbar_map, axes=([0], [0]))
    norm = np.linalg.norm(combined.reshape(-1))
    if norm > 0:
        combined = combined / norm
    return combined, float(norm)


def CollectBranchRecords(node, root_label, path=()):
    if node["branch"] is None:
        return []
    records = [{"root": root_label, "path": tuple(path), "qtilde": float(node["branch"]["qtilde"])}]
    for child_idx, child in enumerate(node["children"]):
        records.extend(CollectBranchRecords(child, root_label, path + (child_idx,)))
    return records


def BuildBranchOrderMap(q_tree, qbar_tree):
    records = CollectBranchRecords(q_tree, "q") + CollectBranchRecords(qbar_tree, "qbar")
    records = sorted(records, key=lambda item: (-item["qtilde"], item["root"], item["path"]))
    order_map = {}
    for idx, record in enumerate(records, start=1):
        order_map[(record["root"], record["path"])] = int(idx)
    return order_map, records


def GetTreeNodeByPath(root_node, path):
    node = root_node
    for child_idx in path:
        node = node["children"][int(child_idx)]
    return node


def CollectTreeCutSpecs(node, root_label, path=()):
    cuts = []
    if node["branch"] is None:
        return cuts

    for child_idx, child in enumerate(node["children"]):
        child_path = tuple(path + (child_idx,))
        cuts.append(
            {
                "root": root_label,
                "path": child_path,
                "parent_pid": int(node["pid"]),
                "child_pid": int(child["pid"]),
                "branch_type": node["branch"]["type"],
                "child_dim": int(color_dimension(child["pid"])),
            }
        )
        cuts.extend(CollectTreeCutSpecs(child, root_label, child_path))
    return cuts


def LeafAxesForSubtree(leaves, root_label, path):
    axes = []
    prefix_length = len(path)
    for leaf in leaves:
        if leaf["root"] != root_label:
            continue
        if leaf["path"][:prefix_length] == path:
            axes.append(int(leaf["axis"]))
    return axes


def AnalyzeFixedHistoryTrees(q_tree, qbar_tree, max_dense_gluons):
    q_leaves = CollectTreeLeaves(q_tree, "q")
    qbar_leaves = CollectTreeLeaves(qbar_tree, "qbar")
    leaves = q_leaves + qbar_leaves
    for axis, leaf in enumerate(leaves):
        leaf["axis"] = int(axis)

    dense_dimension_cap = DenseDimensionCap(max_dense_gluons)
    final_color_dimension = DenseStateDimensionFromLeaves(leaves)

    report = {
        "dense_analyzed": False,
        "dense_skip_reason": None,
        "branch_cut": None,
        "tree_cuts": tuple(),
        "final_color_dimension": int(final_color_dimension),
        "dense_dimension_cap": int(dense_dimension_cap),
        "leaf_count": int(len(leaves)),
    }

    if final_color_dimension > dense_dimension_cap:
        report["dense_skip_reason"] = "dense_dimension_cap"
        return report

    state, norm = BuildEventColorState(q_tree, qbar_tree)
    if norm <= TTN_EPS:
        report["dense_skip_reason"] = "zero_norm_color_state"
        return report

    q_axes = [leaf["axis"] for leaf in leaves if leaf["root"] == "q"]
    report["branch_cut"] = SchmidtDiagnostics(state, q_axes)

    tree_cuts = []
    cut_specs = CollectTreeCutSpecs(q_tree, "q") + CollectTreeCutSpecs(qbar_tree, "qbar")
    for cut in cut_specs:
        axes = LeafAxesForSubtree(leaves, cut["root"], cut["path"])
        cut_diag = SchmidtDiagnostics(state, axes)
        subtree_dimension = 1
        for axis in axes:
            subtree_dimension *= int(leaves[axis]["dim"])
        tree_cuts.append(
            {
                "root": cut["root"],
                "path": cut["path"],
                "parent_pid": cut["parent_pid"],
                "child_pid": cut["child_pid"],
                "branch_type": cut["branch_type"],
                "child_dim": cut["child_dim"],
                "subtree_leaf_count": int(len(axes)),
                "subtree_dimension": int(subtree_dimension),
                "rank": cut_diag["rank"],
                "entropy": cut_diag["entropy"],
                "singular_values": cut_diag["singular_values"],
            }
        )

    report["dense_analyzed"] = True
    report["tree_cuts"] = tuple(tree_cuts)
    return report


def CollectFrontierLeaves(node, root_label, branch_order_map, step, path=()):
    branch_key = (root_label, tuple(path))
    if node["branch"] is None or branch_order_map[branch_key] > step:
        return [
            {
                "root": root_label,
                "path": tuple(path),
                "pid": int(node["pid"]),
                "dim": int(color_dimension(node["pid"])),
            }
        ]
    leaves = []
    for child_idx, child in enumerate(node["children"]):
        leaves.extend(CollectFrontierLeaves(child, root_label, branch_order_map, step, path + (child_idx,)))
    return leaves


def BuildFrontierColorMap(node, root_label, branch_order_map, step, path=()):
    branch_key = (root_label, tuple(path))
    if node["branch"] is None or branch_order_map[branch_key] > step:
        dim = color_dimension(node["pid"])
        return np.eye(dim, dtype=np.complex128)

    child0_map = BuildFrontierColorMap(node["children"][0], root_label, branch_order_map, step, path + (0,))
    child1_map = BuildFrontierColorMap(node["children"][1], root_label, branch_order_map, step, path + (1,))
    local_tensor = BranchColorTensor(node["pid"], node["branch"])

    combined = np.tensordot(local_tensor, child0_map, axes=([1], [0]))
    perm = [0] + list(range(2, combined.ndim)) + [1]
    combined = np.transpose(combined, perm)
    combined = np.tensordot(combined, child1_map, axes=([-1], [0]))
    return combined


def BuildFrontierColorState(q_tree, qbar_tree, branch_order_map, step):
    q_map = BuildFrontierColorMap(q_tree, "q", branch_order_map, step)
    qbar_map = BuildFrontierColorMap(qbar_tree, "qbar", branch_order_map, step)
    combined = np.tensordot(TTN_HARD_SINGLET, q_map, axes=([0], [0]))
    combined = np.tensordot(combined, qbar_map, axes=([0], [0]))
    norm = np.linalg.norm(combined.reshape(-1))
    if norm > 0:
        combined = combined / norm
    return combined, float(norm)


def AnalyzeFrontierSlices(q_tree, qbar_tree, max_dense_gluons):
    branch_order_map, branch_records = BuildBranchOrderMap(q_tree, qbar_tree)
    dense_dimension_cap = DenseDimensionCap(max_dense_gluons)
    total_steps = len(branch_records)
    slice_summaries = []
    ttn_direct = __import__("ttn_direct_frontier")

    for step in range(total_steps + 1):
        leaves = CollectFrontierLeaves(q_tree, "q", branch_order_map, step) + CollectFrontierLeaves(
            qbar_tree, "qbar", branch_order_map, step
        )
        for axis, leaf in enumerate(leaves):
            leaf["axis"] = int(axis)

        frontier_dimension = DenseStateDimensionFromLeaves(leaves)
        payload = {
            "step": int(step),
            "active_line_count": int(len(leaves)),
            "frontier_dimension": int(frontier_dimension),
            "dense_analyzed": False,
            "skip_reason": None,
            "root_entropy": None,
            "root_rank": None,
            "max_chain_entropy": None,
            "max_chain_rank": None,
            "selected_split": None,
            "selected_split_entropy": None,
            "selected_split_rank": None,
            "truncation_tests": tuple(),
            "local_tree_parent_pid": None,
            "local_tree_parent_dim": None,
            "local_tree_entropy": None,
            "local_tree_rank": None,
            "local_tree_discarded_weight": None,
            "local_tree_max_abs_delta": None,
            "local_tree_mean_abs_delta": None,
        }

        if frontier_dimension > dense_dimension_cap:
            payload["skip_reason"] = "dense_dimension_cap"
            slice_summaries.append(payload)
            continue

        state, norm = BuildFrontierColorState(q_tree, qbar_tree, branch_order_map, step)
        if norm <= TTN_EPS:
            payload["skip_reason"] = "zero_norm_color_state"
            slice_summaries.append(payload)
            continue
        exact_correlators = ttn_direct.frontier_pair_correlators(sys.modules[__name__], q_tree, qbar_tree, branch_order_map, step, leaves)

        root_axes = [leaf["axis"] for leaf in leaves if leaf["root"] == "q"]
        root_cut = SchmidtDiagnostics(state, root_axes)
        chain_cuts = []
        for split_idx in range(1, len(leaves)):
            axes_left = list(range(split_idx))
            cut_data = CutSVD(state, axes_left)
            singular_values = cut_data["singular_values"]
            probs = singular_values**2
            probs = probs[probs > TTN_EPS]
            entropy = float(-np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0
            rank = int(np.sum(singular_values > TTN_EPS))
            chain_cuts.append(
                {
                    "split_idx": int(split_idx),
                    "entropy": entropy,
                    "rank": rank,
                    "cut_data": cut_data,
                }
            )

        payload["dense_analyzed"] = True
        payload["root_entropy"] = float(root_cut["entropy"])
        payload["root_rank"] = int(root_cut["rank"])
        if step > 0:
            branch_record = branch_records[step - 1]
            root_node = q_tree if branch_record["root"] == "q" else qbar_tree
            branch_node = GetTreeNodeByPath(root_node, branch_record["path"])
            local_axes = LeafAxesForSubtree(leaves, branch_record["root"], branch_record["path"])
            local_cut_data = CutSVD(state, local_axes)
            local_singular_values = local_cut_data["singular_values"]
            local_probs = local_singular_values**2
            local_probs = local_probs[local_probs > TTN_EPS]
            local_entropy = float(-np.sum(local_probs * np.log(local_probs))) if len(local_probs) > 0 else 0.0
            local_rank = int(np.sum(local_singular_values > TTN_EPS))
            local_parent_dim = int(color_dimension(branch_node["pid"]))
            approx_state, truncation = TruncateStateFromSVD(state, local_cut_data, local_parent_dim)
            approx_correlators = AllPairColorCorrelators(approx_state, leaves)
            local_error_summary = SummarizeCorrelatorErrors(exact_correlators, approx_correlators)
            payload["local_tree_parent_pid"] = int(branch_node["pid"])
            payload["local_tree_parent_dim"] = int(local_parent_dim)
            payload["local_tree_entropy"] = float(local_entropy)
            payload["local_tree_rank"] = int(local_rank)
            payload["local_tree_discarded_weight"] = float(truncation["discarded_weight"])
            payload["local_tree_max_abs_delta"] = float(local_error_summary["max_abs_delta"])
            payload["local_tree_mean_abs_delta"] = float(local_error_summary["mean_abs_delta"])

        if len(chain_cuts) == 0:
            payload["max_chain_entropy"] = 0.0
            payload["max_chain_rank"] = 1
        else:
            payload["max_chain_entropy"] = float(max(cut["entropy"] for cut in chain_cuts))
            payload["max_chain_rank"] = int(max(cut["rank"] for cut in chain_cuts))
            selected_cut = max(chain_cuts, key=lambda cut: (cut["entropy"], cut["rank"], -abs(2 * cut["split_idx"] - len(leaves))))
            payload["selected_split"] = int(selected_cut["split_idx"])
            payload["selected_split_entropy"] = float(selected_cut["entropy"])
            payload["selected_split_rank"] = int(selected_cut["rank"])

            truncation_tests = []
            for chi_target in ttn_truncation_chis:
                approx_state, truncation = TruncateStateFromSVD(state, selected_cut["cut_data"], chi_target)
                if approx_state is None:
                    continue
                approx_correlators = AllPairColorCorrelators(approx_state, leaves)
                error_summary = SummarizeCorrelatorErrors(exact_correlators, approx_correlators)
                truncation_tests.append(
                    {
                        "chi_target": int(truncation["chi_target"]),
                        "full_rank": int(truncation["full_rank"]),
                        "kept_rank": int(truncation["kept_rank"]),
                        "discarded_weight": float(truncation["discarded_weight"]),
                        "kept_norm": float(truncation["kept_norm"]),
                        "pair_count": int(error_summary["pair_count"]),
                        "mean_abs_delta": float(error_summary["mean_abs_delta"]),
                        "rms_delta": float(error_summary["rms_delta"]),
                        "max_abs_delta": float(error_summary["max_abs_delta"]),
                    }
                )
            payload["truncation_tests"] = tuple(truncation_tests)
        slice_summaries.append(payload)

    dense_slices = [entry for entry in slice_summaries if entry["dense_analyzed"]]
    if len(dense_slices) > 0:
        max_chain_entropy = float(max(entry["max_chain_entropy"] for entry in dense_slices))
        max_chain_rank = int(max(entry["max_chain_rank"] for entry in dense_slices))
        max_active_lines = int(max(entry["active_line_count"] for entry in slice_summaries))
        max_frontier_dimension = int(max(entry["frontier_dimension"] for entry in slice_summaries))
    else:
        max_chain_entropy = None
        max_chain_rank = 0
        max_active_lines = int(max(entry["active_line_count"] for entry in slice_summaries)) if len(slice_summaries) else 0
        max_frontier_dimension = int(max(entry["frontier_dimension"] for entry in slice_summaries)) if len(slice_summaries) else 0

    return {
        "slice_summaries": tuple(slice_summaries),
        "summary": {
            "total_slices": int(len(slice_summaries)),
            "dense_slices": int(len(dense_slices)),
            "max_active_lines": int(max_active_lines),
            "max_frontier_dimension": int(max_frontier_dimension),
            "max_chain_entropy": max_chain_entropy,
            "max_chain_rank": int(max_chain_rank),
        },
    }


@lru_cache(maxsize=None)
def BuildFixedHistoryColorState(nq, nqbar):
    psi = TTN_HARD_SINGLET.copy()
    q_axes = 0
    qbar_axes = 0

    for _ in range(nq):
        updated = np.tensordot(TTN_TF, psi, axes=([2], [0]))
        perm = [1, 2] + list(range(3, 3 + q_axes)) + [0] + list(range(3 + q_axes, updated.ndim))
        psi = np.transpose(updated, perm)
        q_axes += 1

    for _ in range(nqbar):
        updated = np.tensordot(TTN_TBAR, psi, axes=([2], [1]))
        perm = [2, 1] + list(range(3, 3 + q_axes)) + list(range(3 + q_axes, updated.ndim)) + [0]
        psi = np.transpose(updated, perm)
        qbar_axes += 1

    norm = np.linalg.norm(psi.reshape(-1))
    if norm > 0:
        psi = psi / norm
    return psi


def ReshapeStateForCut(state, axes_a):
    axes_a = sorted(set(axes_a))
    axes_b = [axis for axis in range(state.ndim) if axis not in axes_a]
    perm = axes_a + axes_b
    permuted = np.transpose(state, perm)
    left_dim = int(np.prod([state.shape[axis] for axis in axes_a], dtype=np.int64))
    right_dim = int(np.prod([state.shape[axis] for axis in axes_b], dtype=np.int64))
    matrix = permuted.reshape(left_dim, right_dim)
    inverse_perm = tuple(int(idx) for idx in np.argsort(perm))
    return {
        "axes_a": tuple(axes_a),
        "axes_b": tuple(axes_b),
        "perm": tuple(perm),
        "inverse_perm": inverse_perm,
        "left_dim": int(left_dim),
        "right_dim": int(right_dim),
        "permuted_shape": tuple(int(x) for x in permuted.shape),
        "matrix": matrix,
    }


def CutSVD(state, axes_a):
    cut_data = ReshapeStateForCut(state, axes_a)
    singular_values = np.linalg.svd(cut_data["matrix"], full_matrices=False, compute_uv=False)
    cut_data["singular_values"] = singular_values
    return cut_data


def SchmidtDiagnostics(state, axes_a):
    singular_values = CutSVD(state, axes_a)["singular_values"]
    probs = singular_values**2
    probs = probs[probs > TTN_EPS]
    entropy = 0.0
    if len(probs) > 0:
        entropy = float(-np.sum(probs * np.log(probs)))
    kept = tuple(float(value) for value in singular_values if value > TTN_EPS)
    return {"rank": len(kept), "entropy": entropy, "singular_values": kept}


@lru_cache(maxsize=None)
def ColorGeneratorsForPid(pid):
    pid = int(pid)
    if pid == 21:
        return TTN_TG
    if pid > 0:
        return TTN_TF
    return TTN_TBAR


@lru_cache(maxsize=None)
def PairColorOperator(pid_i, pid_j):
    gens_i = ColorGeneratorsForPid(pid_i)
    gens_j = ColorGeneratorsForPid(pid_j)
    dim_i = gens_i.shape[1]
    dim_j = gens_j.shape[1]
    operator = np.zeros((dim_i * dim_j, dim_i * dim_j), dtype=np.complex128)
    for color_idx in range(8):
        operator = operator + np.kron(gens_i[color_idx], gens_j[color_idx])
    return operator


def PairColorCorrelator(state, leaves, axis_i, axis_j):
    if axis_i == axis_j:
        raise ValueError("PairColorCorrelator requires two distinct axes")
    axes = [int(axis_i), int(axis_j)]
    cut_data = ReshapeStateForCut(state, axes)
    pair_density = cut_data["matrix"] @ np.conjugate(cut_data["matrix"]).T
    operator = PairColorOperator(leaves[axis_i]["pid"], leaves[axis_j]["pid"])
    value = np.trace(pair_density @ operator)
    return float(np.real(value))


def AllPairColorCorrelators(state, leaves):
    correlators = []
    for axis_i in range(len(leaves)):
        for axis_j in range(axis_i + 1, len(leaves)):
            correlators.append(
                {
                    "pair": (int(axis_i), int(axis_j)),
                    "value": PairColorCorrelator(state, leaves, axis_i, axis_j),
                }
            )
    return tuple(correlators)


def TruncateStateFromSVD(state, cut_data, chi_target):
    matrix = cut_data["matrix"]
    U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
    full_rank = int(np.sum(singular_values > TTN_EPS))
    kept_rank = min(int(chi_target), full_rank)
    if kept_rank <= 0:
        return None, {
            "chi_target": int(chi_target),
            "full_rank": int(full_rank),
            "kept_rank": 0,
            "discarded_weight": 1.0,
            "kept_norm": 0.0,
        }

    approx_matrix = (U[:, :kept_rank] * singular_values[:kept_rank]) @ Vh[:kept_rank, :]
    kept_norm_sq = float(np.sum(singular_values[:kept_rank] ** 2))
    discarded_weight = float(np.sum(singular_values[kept_rank:] ** 2))
    if kept_norm_sq > TTN_EPS:
        approx_matrix = approx_matrix / np.sqrt(kept_norm_sq)
    approx_permuted = approx_matrix.reshape(cut_data["permuted_shape"])
    approx_state = np.transpose(approx_permuted, cut_data["inverse_perm"])
    return approx_state, {
        "chi_target": int(chi_target),
        "full_rank": int(full_rank),
        "kept_rank": int(kept_rank),
        "discarded_weight": float(discarded_weight),
        "kept_norm": float(np.sqrt(max(kept_norm_sq, 0.0))),
    }


def SummarizeCorrelatorErrors(exact_correlators, approx_correlators):
    exact_values = [float(entry["value"]) for entry in exact_correlators]
    approx_values = [float(entry["value"]) for entry in approx_correlators]
    deltas = [abs(exact_values[idx] - approx_values[idx]) for idx in range(len(exact_values))]
    if len(deltas) == 0:
        return {"pair_count": 0, "mean_abs_delta": 0.0, "rms_delta": 0.0, "max_abs_delta": 0.0}
    mean_abs_delta = float(sum(deltas) / len(deltas))
    rms_delta = float(np.sqrt(sum(delta * delta for delta in deltas) / len(deltas)))
    max_abs_delta = float(max(deltas))
    return {
        "pair_count": int(len(deltas)),
        "mean_abs_delta": mean_abs_delta,
        "rms_delta": rms_delta,
        "max_abs_delta": max_abs_delta,
    }


@lru_cache(maxsize=None)
def AnalyzeFixedHistoryCounts(nq, nqbar, max_dense_gluons):
    if nq + nqbar > max_dense_gluons:
        return {
            "nq": nq,
            "nqbar": nqbar,
            "dense_analyzed": False,
            "branch_cut": None,
            "q_chain_cuts": tuple(),
            "qbar_chain_cuts": tuple(),
        }

    state = BuildFixedHistoryColorState(nq, nqbar)
    q_gluon_start = 2
    qbar_gluon_start = 2 + nq

    branch_axes = [0] + list(range(q_gluon_start, q_gluon_start + nq))
    branch_cut = SchmidtDiagnostics(state, branch_axes)

    q_chain_cuts = []
    for cut_idx in range(nq + 1):
        axes = [0] + list(range(q_gluon_start + cut_idx, q_gluon_start + nq))
        q_chain_cuts.append(SchmidtDiagnostics(state, axes))

    qbar_chain_cuts = []
    for cut_idx in range(nqbar + 1):
        axes = [1] + list(range(qbar_gluon_start + cut_idx, qbar_gluon_start + nqbar))
        qbar_chain_cuts.append(SchmidtDiagnostics(state, axes))

    return {
        "nq": nq,
        "nqbar": nqbar,
        "dense_analyzed": True,
        "branch_cut": branch_cut,
        "q_chain_cuts": tuple(q_chain_cuts),
        "qbar_chain_cuts": tuple(qbar_chain_cuts),
    }


def AnalyzeJetHistories(jet_histories, max_dense_gluons):
    q_histories = [history for history in jet_histories if history["branch_type"] == "q"]
    qbar_histories = [history for history in jet_histories if history["branch_type"] == "qbar"]
    if len(q_histories) != 1 or len(qbar_histories) != 1:
        return None
    branch_counts = {
        "q->qg": q_histories[0]["branch_counts"]["q->qg"] + qbar_histories[0]["branch_counts"]["q->qg"],
        "g->gg": q_histories[0]["branch_counts"]["g->gg"] + qbar_histories[0]["branch_counts"]["g->gg"],
        "g->qqbar": q_histories[0]["branch_counts"]["g->qqbar"] + qbar_histories[0]["branch_counts"]["g->qqbar"],
    }
    exact_applicable = (q_histories[0]["has_gluon_branching"] is False) and (qbar_histories[0]["has_gluon_branching"] is False)
    tree_report = AnalyzeFixedHistoryTrees(q_histories[0]["tree"], qbar_histories[0]["tree"], max_dense_gluons)
    frontier_report = AnalyzeFrontierSlices(q_histories[0]["tree"], qbar_histories[0]["tree"], max_dense_gluons)
    if exact_applicable:
        nq = len(q_histories[0]["emissions"])
        nqbar = len(qbar_histories[0]["emissions"])
        q_chain_report = AnalyzeFixedHistoryCounts(nq, nqbar, max_dense_gluons)
    else:
        nq = None
        nqbar = None
        q_chain_report = {
            "nq": None,
            "nqbar": None,
            "dense_analyzed": False,
            "branch_cut": None,
            "q_chain_cuts": tuple(),
            "qbar_chain_cuts": tuple(),
        }
    report = {
        "nq": nq,
        "nqbar": nqbar,
        "dense_analyzed": bool(tree_report["dense_analyzed"]),
        "dense_skip_reason": tree_report.get("dense_skip_reason"),
        "branch_cut": tree_report["branch_cut"],
        "q_chain_cuts": tuple(q_chain_report["q_chain_cuts"]),
        "qbar_chain_cuts": tuple(q_chain_report["qbar_chain_cuts"]),
        "tree_cuts": tuple(tree_report["tree_cuts"]),
        "q_chain_dense_analyzed": bool(q_chain_report["dense_analyzed"]),
        "final_color_dimension": int(tree_report["final_color_dimension"]),
        "dense_dimension_cap": int(tree_report["dense_dimension_cap"]),
        "leaf_count": int(tree_report["leaf_count"]),
        "frontier_slices": tuple(frontier_report["slice_summaries"]),
        "frontier_summary": dict(frontier_report["summary"]),
    }
    report["exact_q_chain_applicable"] = exact_applicable
    report["branch_counts"] = branch_counts
    report["q_root_branch_counts"] = q_histories[0]["branch_counts"]
    report["qbar_root_branch_counts"] = qbar_histories[0]["branch_counts"]
    report["final_parton_count"] = len(q_histories[0]["momenta"]) + len(qbar_histories[0]["momenta"])
    return report


def SerializeCutDiagnostics(cut):
    if cut is None:
        return None
    return {
        "rank": int(cut["rank"]),
        "entropy": float(cut["entropy"]),
        "singular_values": [float(value) for value in cut["singular_values"]],
    }


def SerializeTreeCutDiagnostics(cut):
    if cut is None:
        return None
    payload = SerializeCutDiagnostics(cut)
    payload.update(
        {
            "root": str(cut["root"]),
            "path": [int(entry) for entry in cut["path"]],
            "parent_pid": int(cut["parent_pid"]),
            "child_pid": int(cut["child_pid"]),
            "branch_type": str(cut["branch_type"]),
            "child_dim": int(cut["child_dim"]),
            "subtree_leaf_count": int(cut["subtree_leaf_count"]),
            "subtree_dimension": int(cut["subtree_dimension"]),
        }
    )
    return payload


def SerializeFrontierTruncationDiagnostics(entry):
    return {
        "chi_target": int(entry["chi_target"]),
        "full_rank": int(entry["full_rank"]),
        "kept_rank": int(entry["kept_rank"]),
        "discarded_weight": float(entry["discarded_weight"]),
        "kept_norm": float(entry["kept_norm"]),
        "pair_count": int(entry["pair_count"]),
        "mean_abs_delta": float(entry["mean_abs_delta"]),
        "rms_delta": float(entry["rms_delta"]),
        "max_abs_delta": float(entry["max_abs_delta"]),
    }


def SerializeFrontierSliceDiagnostics(entry):
    return {
        "step": int(entry["step"]),
        "active_line_count": int(entry["active_line_count"]),
        "frontier_dimension": int(entry["frontier_dimension"]),
        "dense_analyzed": bool(entry["dense_analyzed"]),
        "skip_reason": entry.get("skip_reason"),
        "root_entropy": None if entry["root_entropy"] is None else float(entry["root_entropy"]),
        "root_rank": None if entry["root_rank"] is None else int(entry["root_rank"]),
        "max_chain_entropy": None if entry["max_chain_entropy"] is None else float(entry["max_chain_entropy"]),
        "max_chain_rank": None if entry["max_chain_rank"] is None else int(entry["max_chain_rank"]),
        "selected_split": None if entry["selected_split"] is None else int(entry["selected_split"]),
        "selected_split_entropy": None
        if entry["selected_split_entropy"] is None
        else float(entry["selected_split_entropy"]),
        "selected_split_rank": None if entry["selected_split_rank"] is None else int(entry["selected_split_rank"]),
        "truncation_tests": [SerializeFrontierTruncationDiagnostics(test) for test in entry.get("truncation_tests", tuple())],
        "local_tree_parent_pid": None if entry["local_tree_parent_pid"] is None else int(entry["local_tree_parent_pid"]),
        "local_tree_parent_dim": None if entry["local_tree_parent_dim"] is None else int(entry["local_tree_parent_dim"]),
        "local_tree_entropy": None if entry["local_tree_entropy"] is None else float(entry["local_tree_entropy"]),
        "local_tree_rank": None if entry["local_tree_rank"] is None else int(entry["local_tree_rank"]),
        "local_tree_discarded_weight": None
        if entry["local_tree_discarded_weight"] is None
        else float(entry["local_tree_discarded_weight"]),
        "local_tree_max_abs_delta": None
        if entry["local_tree_max_abs_delta"] is None
        else float(entry["local_tree_max_abs_delta"]),
        "local_tree_mean_abs_delta": None
        if entry["local_tree_mean_abs_delta"] is None
        else float(entry["local_tree_mean_abs_delta"]),
    }


def SerializeEventTTNReport(event_index, report):
    if report is None:
        return {"event_index": int(event_index), "analyzed": False}
    return {
        "event_index": int(event_index),
        "analyzed": True,
        "nq": None if report["nq"] is None else int(report["nq"]),
        "nqbar": None if report["nqbar"] is None else int(report["nqbar"]),
        "exact_q_chain_applicable": bool(report.get("exact_q_chain_applicable", False)),
        "branch_counts": report.get("branch_counts", {}),
        "final_parton_count": int(report.get("final_parton_count", 0)),
        "final_color_dimension": int(report.get("final_color_dimension", 0)),
        "dense_dimension_cap": int(report.get("dense_dimension_cap", 0)),
        "leaf_count": int(report.get("leaf_count", 0)),
        "dense_analyzed": bool(report["dense_analyzed"]),
        "dense_skip_reason": report.get("dense_skip_reason"),
        "q_chain_dense_analyzed": bool(report.get("q_chain_dense_analyzed", False)),
        "branch_cut": SerializeCutDiagnostics(report["branch_cut"]),
        "q_chain_cuts": [SerializeCutDiagnostics(cut) for cut in report["q_chain_cuts"]],
        "qbar_chain_cuts": [SerializeCutDiagnostics(cut) for cut in report["qbar_chain_cuts"]],
        "tree_cuts": [SerializeTreeCutDiagnostics(cut) for cut in report.get("tree_cuts", tuple())],
        "frontier_summary": report.get("frontier_summary", {}),
        "frontier_slices": [SerializeFrontierSliceDiagnostics(entry) for entry in report.get("frontier_slices", tuple())],
    }


def Percentile(values, quantile):
    if len(values) == 0:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def SummarizeCutCollection(cuts):
    if len(cuts) == 0:
        return {"count": 0, "mean_entropy": None, "p95_entropy": None, "max_entropy": None, "max_rank": 0}
    entropies = [float(cut["entropy"]) for cut in cuts]
    ranks = [int(cut["rank"]) for cut in cuts]
    return {
        "count": int(len(cuts)),
        "mean_entropy": float(sum(entropies) / len(entropies)),
        "p95_entropy": Percentile(entropies, 0.95),
        "max_entropy": float(max(entropies)),
        "max_rank": int(max(ranks)),
    }


def SummarizeTruncationCollection(entries):
    if len(entries) == 0:
        return {
            "count": 0,
            "mean_discarded_weight": None,
            "p95_discarded_weight": None,
            "max_discarded_weight": None,
            "mean_max_abs_delta": None,
            "p95_max_abs_delta": None,
            "max_max_abs_delta": None,
            "mean_rms_delta": None,
            "p95_rms_delta": None,
            "max_rms_delta": None,
        }
    discarded = [float(entry["discarded_weight"]) for entry in entries]
    max_abs = [float(entry["max_abs_delta"]) for entry in entries]
    rms = [float(entry["rms_delta"]) for entry in entries]
    return {
        "count": int(len(entries)),
        "mean_discarded_weight": float(sum(discarded) / len(discarded)),
        "p95_discarded_weight": Percentile(discarded, 0.95),
        "max_discarded_weight": float(max(discarded)),
        "mean_max_abs_delta": float(sum(max_abs) / len(max_abs)),
        "p95_max_abs_delta": Percentile(max_abs, 0.95),
        "max_max_abs_delta": float(max(max_abs)),
        "mean_rms_delta": float(sum(rms) / len(rms)),
        "p95_rms_delta": Percentile(rms, 0.95),
        "max_rms_delta": float(max(rms)),
    }


def SummarizeLocalTreeCompression(entries):
    if len(entries) == 0:
        return {
            "count": 0,
            "mean_entropy": None,
            "p95_entropy": None,
            "max_entropy": None,
            "max_rank": 0,
            "mean_discarded_weight": None,
            "max_discarded_weight": None,
            "mean_max_abs_delta": None,
            "max_max_abs_delta": None,
        }
    entropies = [float(entry["local_tree_entropy"]) for entry in entries]
    ranks = [int(entry["local_tree_rank"]) for entry in entries]
    discarded = [float(entry["local_tree_discarded_weight"]) for entry in entries]
    max_abs = [float(entry["local_tree_max_abs_delta"]) for entry in entries]
    return {
        "count": int(len(entries)),
        "mean_entropy": float(sum(entropies) / len(entropies)),
        "p95_entropy": Percentile(entropies, 0.95),
        "max_entropy": float(max(entropies)),
        "max_rank": int(max(ranks)),
        "mean_discarded_weight": float(sum(discarded) / len(discarded)),
        "max_discarded_weight": float(max(discarded)),
        "mean_max_abs_delta": float(sum(max_abs) / len(max_abs)),
        "max_max_abs_delta": float(max(max_abs)),
    }


def BuildTTNSummary(ttn_reports, max_dense_gluons):
    usable_reports = [report for report in ttn_reports if report is not None]
    grouped = {}
    skipped = 0
    q_chain_skipped = 0
    total_branch_counts = {"q->qg": 0, "g->gg": 0, "g->qqbar": 0}
    exact_applicable_count = 0
    dense_tree_event_count = 0
    dense_tree_with_gluon_branching_count = 0
    branch_cuts = []
    tree_cuts = []
    quark_tree_cuts = []
    gluon_tree_cuts = []
    frontier_dense_slices = []
    frontier_event_maxima = []
    frontier_dense_gluon_event_count = 0
    frontier_truncation_by_chi = {int(chi): [] for chi in ttn_truncation_chis}
    local_tree_entries = []
    local_tree_quark_entries = []
    local_tree_gluon_entries = []
    for report in usable_reports:
        for key in total_branch_counts:
            total_branch_counts[key] += report.get("branch_counts", {}).get(key, 0)
        if report["dense_analyzed"]:
            dense_tree_event_count += 1
            if report.get("branch_counts", {}).get("g->gg", 0) > 0 or report.get("branch_counts", {}).get("g->qqbar", 0) > 0:
                dense_tree_with_gluon_branching_count += 1
            if report.get("branch_cut") is not None:
                branch_cuts.append(report["branch_cut"])
            for cut in report.get("tree_cuts", tuple()):
                tree_cuts.append(cut)
                if cut["child_dim"] == 8:
                    gluon_tree_cuts.append(cut)
                else:
                    quark_tree_cuts.append(cut)
        else:
            skipped += 1
        frontier_summary = report.get("frontier_summary", {})
        if frontier_summary.get("dense_slices", 0) > 0:
            frontier_event_maxima.append(
                {
                    "entropy": frontier_summary.get("max_chain_entropy"),
                    "rank": frontier_summary.get("max_chain_rank", 0),
                }
            )
            if report.get("branch_counts", {}).get("g->gg", 0) > 0 or report.get("branch_counts", {}).get("g->qqbar", 0) > 0:
                frontier_dense_gluon_event_count += 1
        for entry in report.get("frontier_slices", tuple()):
            if entry.get("dense_analyzed", False):
                frontier_dense_slices.append(
                    {
                        "entropy": entry["max_chain_entropy"],
                        "rank": entry["max_chain_rank"],
                    }
                )
                if entry.get("local_tree_parent_dim") is not None:
                    local_tree_entries.append(entry)
                    if int(entry["local_tree_parent_dim"]) == 8:
                        local_tree_gluon_entries.append(entry)
                    else:
                        local_tree_quark_entries.append(entry)
                for truncation in entry.get("truncation_tests", tuple()):
                    chi_target = int(truncation["chi_target"])
                    if chi_target not in frontier_truncation_by_chi:
                        frontier_truncation_by_chi[chi_target] = []
                    frontier_truncation_by_chi[chi_target].append(truncation)
        if report.get("exact_q_chain_applicable", False):
            exact_applicable_count += 1
            key = (report["nq"], report["nqbar"])
            if key not in grouped:
                grouped[key] = {"count": 0, "report": report}
            grouped[key]["count"] += 1
        if report.get("exact_q_chain_applicable", False) and report.get("q_chain_dense_analyzed", False) is False:
            q_chain_skipped += 1

    grouped_rows = []
    for key in sorted(grouped.keys()):
        nq, nqbar = key
        count = grouped[key]["count"]
        report = grouped[key]["report"]
        row = {
            "nq": int(nq),
            "nqbar": int(nqbar),
            "events": int(count),
            "dense_analyzed": bool(report.get("q_chain_dense_analyzed", False)),
        }
        if report.get("q_chain_dense_analyzed", False) and report["branch_cut"] is not None:
            max_q_entropy = max([cut["entropy"] for cut in report["q_chain_cuts"]]) if len(report["q_chain_cuts"]) else 0.0
            max_qbar_entropy = max([cut["entropy"] for cut in report["qbar_chain_cuts"]]) if len(report["qbar_chain_cuts"]) else 0.0
            ranks = [report["branch_cut"]["rank"]]
            ranks += [cut["rank"] for cut in report["q_chain_cuts"]]
            ranks += [cut["rank"] for cut in report["qbar_chain_cuts"]]
            row["branch_entropy"] = float(report["branch_cut"]["entropy"])
            row["max_q_chain_entropy"] = float(max_q_entropy)
            row["max_qbar_chain_entropy"] = float(max_qbar_entropy)
            row["max_rank"] = int(max(ranks))
        else:
            row["branch_entropy"] = None
            row["max_q_chain_entropy"] = None
            row["max_qbar_chain_entropy"] = None
            row["max_rank"] = 3
        grouped_rows.append(row)

    return {
        "max_dense_gluons": int(max_dense_gluons),
        "dense_dimension_cap": int(DenseDimensionCap(max_dense_gluons)),
        "usable_event_count": int(len(usable_reports)),
        "dense_tree_event_count": int(dense_tree_event_count),
        "dense_tree_with_gluon_branching_event_count": int(dense_tree_with_gluon_branching_count),
        "exact_q_chain_event_count": int(exact_applicable_count),
        "skipped_dense_event_count": int(skipped),
        "q_chain_skipped_dense_event_count": int(q_chain_skipped),
        "total_branch_counts": total_branch_counts,
        "branch_cut_stats": SummarizeCutCollection(branch_cuts),
        "tree_cut_stats": {
            "all": SummarizeCutCollection(tree_cuts),
            "quark_child": SummarizeCutCollection(quark_tree_cuts),
            "gluon_child": SummarizeCutCollection(gluon_tree_cuts),
        },
        "frontier_slice_stats": SummarizeCutCollection(frontier_dense_slices),
        "frontier_event_stats": SummarizeCutCollection(frontier_event_maxima),
        "frontier_dense_gluon_event_count": int(frontier_dense_gluon_event_count),
        "frontier_truncation_stats": {
            str(int(chi)): SummarizeTruncationCollection(frontier_truncation_by_chi.get(int(chi), []))
            for chi in sorted(frontier_truncation_by_chi.keys())
        },
        "local_tree_replay_stats": {
            "all": SummarizeLocalTreeCompression(local_tree_entries),
            "quark_parent": SummarizeLocalTreeCompression(local_tree_quark_entries),
            "gluon_parent": SummarizeLocalTreeCompression(local_tree_gluon_entries),
        },
        "grouped_rows": grouped_rows,
    }


def WriteTTNReport(reportfile, metadata, summary, serialized_events):
    payload = {
        "metadata": metadata,
        "summary": summary,
        "events": serialized_events,
    }
    with open(reportfile, "w") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


def SummarizeTTNReports(ttn_reports, max_dense_gluons):
    summary = BuildTTNSummary(ttn_reports, max_dense_gluons)
    usable_reports = [report for report in ttn_reports if report is not None]
    if len(usable_reports) == 0:
        print("\nNo TTN-compatible qqbar histories were found.")
        return summary

    print("\nFixed-history TTN summary")
    print(
        "Dense color-state analysis is exact for events with final color dimension <=",
        summary["dense_dimension_cap"],
        "(equivalent to qqbar +",
        max_dense_gluons,
        "gluons).",
    )
    print("Total accepted branchings:", summary["total_branch_counts"])
    print(
        "Exact dense tree-state analysis covered",
        summary["dense_tree_event_count"],
        "out of",
        summary["usable_event_count"],
        "event(s), including",
        summary["dense_tree_with_gluon_branching_event_count"],
        "with at least one gluon branching.",
    )

    branch_stats = summary["branch_cut_stats"]
    if branch_stats["count"] > 0:
        print(
            "Root q|qbar cut: mean S =",
            f"{branch_stats['mean_entropy']:.6f},",
            "p95 S =",
            f"{branch_stats['p95_entropy']:.6f},",
            "max S =",
            f"{branch_stats['max_entropy']:.6f},",
            "max rank =",
            branch_stats["max_rank"],
        )

    tree_stats_table = PrettyTable(["subtree cut", "count", "mean S", "p95 S", "max S", "max rank"])
    for label, stats in [("all", summary["tree_cut_stats"]["all"]), ("quark child", summary["tree_cut_stats"]["quark_child"]), ("gluon child", summary["tree_cut_stats"]["gluon_child"])]:
        if stats["count"] == 0:
            tree_stats_table.add_row([label, 0, "-", "-", "-", 0])
        else:
            tree_stats_table.add_row(
                [
                    label,
                    stats["count"],
                    f"{stats['mean_entropy']:.6f}",
                    f"{stats['p95_entropy']:.6f}",
                    f"{stats['max_entropy']:.6f}",
                    stats["max_rank"],
                ]
            )
    print(tree_stats_table)

    frontier_slice_stats = summary["frontier_slice_stats"]
    frontier_event_stats = summary["frontier_event_stats"]
    if frontier_slice_stats["count"] > 0:
        print(
            "Frontier slices (active lines in DFS order): mean max-chain S =",
            f"{frontier_slice_stats['mean_entropy']:.6f},",
            "p95 =",
            f"{frontier_slice_stats['p95_entropy']:.6f},",
            "max =",
            f"{frontier_slice_stats['max_entropy']:.6f},",
            "max rank =",
            frontier_slice_stats["max_rank"],
        )
    if frontier_event_stats["count"] > 0:
        print(
            "Per-event frontier maxima were available for",
            frontier_event_stats["count"],
            "event(s), including",
            summary["frontier_dense_gluon_event_count"],
            "with gluon branching.",
        )

    local_tree_table = PrettyTable(["local replay", "count", "mean S", "max rank", "max disc. wt.", "max |d<Ti.Tj>|"])
    has_local_tree_rows = False
    for label, stats in [
        ("all", summary["local_tree_replay_stats"]["all"]),
        ("quark parent", summary["local_tree_replay_stats"]["quark_parent"]),
        ("gluon parent", summary["local_tree_replay_stats"]["gluon_parent"]),
    ]:
        if stats["count"] == 0:
            continue
        has_local_tree_rows = True
        local_tree_table.add_row(
            [
                label,
                stats["count"],
                f"{stats['mean_entropy']:.6f}",
                stats["max_rank"],
                f"{stats['max_discarded_weight']:.6e}",
                f"{stats['max_max_abs_delta']:.6e}",
            ]
        )
    if has_local_tree_rows:
        print("Tree-local compress-after-emit replay on the parent bond:")
        print(local_tree_table)

    truncation_table = PrettyTable(["chi", "tests", "mean disc. wt.", "p95 disc. wt.", "p95 max |d<Ti.Tj>|", "max |d<Ti.Tj>|"])
    has_truncation_rows = False
    for chi_key in sorted(summary["frontier_truncation_stats"].keys(), key=int):
        stats = summary["frontier_truncation_stats"][chi_key]
        if stats["count"] == 0:
            continue
        has_truncation_rows = True
        truncation_table.add_row(
            [
                chi_key,
                stats["count"],
                f"{stats['mean_discarded_weight']:.6e}",
                f"{stats['p95_discarded_weight']:.6e}",
                f"{stats['p95_max_abs_delta']:.6e}",
                f"{stats['max_max_abs_delta']:.6e}",
            ]
        )
    if has_truncation_rows:
        print("Frontier truncation check on selected max-entropy chain cuts:")
        print(truncation_table)

    print("Pure q->qg chain subset:")

    tbl = PrettyTable(["n_q", "n_qbar", "events", "dense?", "S(q-branch)", "max S(q-chain)", "max S(qbar-chain)", "max rank"])
    for row in summary["grouped_rows"]:
        if row["dense_analyzed"] and row["branch_entropy"] is not None:
            tbl.add_row(
                [
                    row["nq"],
                    row["nqbar"],
                    row["events"],
                    "yes",
                    f"{row['branch_entropy']:.6f}",
                    f"{row['max_q_chain_entropy']:.6f}",
                    f"{row['max_qbar_chain_entropy']:.6f}",
                    row["max_rank"],
                ]
            )
        else:
            tbl.add_row([row["nq"], row["nqbar"], row["events"], "no", "-", "-", "-", "<= 3"])
    print(tbl)
    if summary["exact_q_chain_event_count"] < summary["usable_event_count"]:
        print(
            "Exact q->qg chain TTN analysis was applicable to",
            summary["exact_q_chain_event_count"],
            "out of",
            summary["usable_event_count"],
            "event(s).",
        )
    if summary["q_chain_skipped_dense_event_count"] > 0:
        print(
            "Skipped dense SVDs for",
            summary["q_chain_skipped_dense_event_count"],
            "pure q->qg chain event(s) because the exact color tensor would exceed the gluon cap.",
        )
    non_dense_tree = summary["usable_event_count"] - summary["dense_tree_event_count"]
    if non_dense_tree > 0:
        print(
            "Skipped dense tree-state analysis for",
            non_dense_tree,
            "event(s) because the full final-state color tensor exceeded the dense cap.",
        )
    return summary


##########################
# Evolution begins here! #
##########################


def main(argv=None):
    configure_from_args(argv)

    alphaS_over = get_alphaS_over(Qc)
    if debug:
        print("alphaS overestimate=", alphaS_over)

    print("Showering", inputfile)
    events, weights, multiweights = readlhefile(inputfile)

    showeredEvents = []
    ttnReports = []
    serializedEventReports = []

    max_events = min(len(events), Nshower)
    for i, particles in enumerate(tqdm(events[:max_events])):
        showeredParticles, showeredJets, jetHistories = Shower(particles, pTmin, alphaS_over)
        showeredParticles = GlobalMomCons(showeredParticles, showeredJets)
        if debug is True or printevents is True:
            PrintMomenta(showeredParticles)
            print("Momentum conservation check AFTER=", CheckMomentumConservation(showeredParticles), "\n")
            for history in jetHistories:
                print("Jet history:", history["branch_type"], "with", len(history["emissions"]), "emission(s)")
                PrintEmissions(history["emissions"])
        showeredEvents.append(showeredParticles)
        event_report = AnalyzeJetHistories(jetHistories, ttn_max_gluons)
        ttnReports.append(event_report)
        serializedEventReports.append(SerializeEventTTNReport(i, event_report))

    summary = SummarizeTTNReports(ttnReports, ttn_max_gluons)
    metadata = {
        "inputfile": inputfile,
        "outputfile": outputfile,
        "ttn_reportfile": ttn_reportfile,
        "requested_events": int(Nshower),
        "processed_events": int(max_events),
        "Qc": float(Qc),
        "pTmin": float(pTmin),
        "ttn_max_gluons": int(ttn_max_gluons),
        "ttn_truncation_chis": [int(value) for value in ttn_truncation_chis],
        "skip_output": bool(skip_output),
    }
    WriteTTNReport(ttn_reportfile, metadata, summary, serializedEventReports)
    print("TTN report written to", ttn_reportfile)

    if skip_output is False:
        sigma = 1.2
        error = 0.2
        ECM = 206
        outlhe = outputfile.replace(".hepmc", "_pyr_ttn.lhe")
        fout = init_lhe(outlhe, sigma, error, ECM)
        write_lhe(fout, showeredEvents, ECM**2, debug)
        finalize_lhe(fout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rotations, Lorentz boosts and global recoil for the massless ee tutorials.

Four-vectors use (px, py, pz, E); particle records start with (PDG id, status).
The common momentum rescaling follows section 6.4.2 of arXiv:0803.0883.
"""
import numpy as np
from scipy.optimize import brentq


class KinematicsError(ValueError):
    """The generated shower cannot be reconstructed at the available energy."""


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("A direction must be finite and nonzero")
    return vector / norm


def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1., 1.))


def GetRotationMatrixAB(a, b):
    a, b = unit_vector(a), unit_vector(b)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = np.clip(np.dot(a, b), -1., 1.)
    if sine < 1.e-14:
        if cosine > 0:
            return np.eye(3)
        axis = unit_vector(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
        return 2. * np.outer(axis, axis) - np.eye(3)
    x, y, z = cross / sine
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + sine * skew + (1. - cosine) * (skew @ skew)


def rotate(p, matrix):
    result = list(p)
    result[2:5] = matrix @ np.asarray(p[2:5])
    return result


def RotateMomentaLab(p, momenta):
    matrix = GetRotationMatrixAB([0., 0., 1.], p[2:5])
    return [rotate(momentum, matrix) for momentum in momenta]


def boost(fourvector, betavec):
    p = np.asarray(fourvector, dtype=float)
    beta = np.asarray(betavec, dtype=float)
    b2 = np.dot(beta, beta)
    if not np.all(np.isfinite(p)) or not np.isfinite(b2) or b2 >= 1.:
        raise KinematicsError("A Lorentz boost requires finite momenta and |beta| < 1")
    if b2 == 0.:
        return p.copy()
    gamma = 1. / np.sqrt(1. - b2)
    bp = np.dot(beta, p[:3])
    # gamma**2 / (gamma + 1) = (gamma - 1) / beta**2, stable at beta=0.
    spatial = p[:3] + (gamma**2 / (gamma + 1.) * bp - gamma * p[3]) * beta
    return np.r_[spatial, gamma * (p[3] - bp)]


def mass_squared(p):
    value = (p[3] - np.linalg.norm(p[:3])) * (p[3] + np.linalg.norm(p[:3]))
    if not np.isfinite(value) or value < -1.e-10 * max(1., p[3]**2):
        raise KinematicsError("A shower jet has nonfinite or spacelike momentum")
    return max(0., value)


def getBoostBeta(k, newq, oldp):
    q = np.linalg.norm(newq[:3])
    kp = k * np.linalg.norm(oldp[:3])
    energy = np.sqrt(kp**2 + mass_squared(newq))
    # Match the positive light-cone components. This also works for a
    # massless jet and allows signed boosts when a jet must gain energy.
    initial_plus, target_plus = newq[3] + q, energy + kp
    if initial_plus <= 0. or target_plus <= 0.:
        raise KinematicsError("Cannot boost a zero-energy jet")
    beta = np.tanh(np.log(initial_plus / target_plus))
    return beta * unit_vector(oldp[:3])


def CheckMomentumConservation(momenta):
    return np.sum([p[2:5] for p in momenta if p[1] == 1], axis=0)


def GlobalMomCons(particles, jets):
    if not jets:
        raise ValueError("No shower progenitors found")
    parents = np.asarray([jet[0][2:6] for jet in jets], dtype=float)
    target = parents.sum(axis=0)
    if np.linalg.norm(target[:3]) > 1.e-7 * target[3]:
        raise ValueError("The tutorial requires the hard event in its center-of-mass frame")
    totals = [np.sum([p[2:6] for p in jet[1]], axis=0) for jet in jets]
    masses2 = np.array([mass_squared(p) for p in totals])
    p2 = np.sum(parents[:, :3]**2, axis=1)
    if np.sqrt(masses2).sum() >= target[3]:
        raise KinematicsError("Shower jet masses exceed the hard-event energy")

    def balance(k):
        return np.sqrt(k*k*p2 + masses2).sum() - target[3]

    upper = 1.
    while balance(upper) < 0.:
        upper *= 2.
    k = brentq(balance, 0., upper, xtol=1.e-14)
    result = [list(p) for p in particles if p[1] == -1]
    for (parent, daughters), total in zip(jets, totals):
        norm = np.linalg.norm(total[:3])
        rotation = GetRotationMatrixAB(total[:3], parent[2:5]) if norm > 1.e-14 else np.eye(3)
        beta = getBoostBeta(k, total, np.asarray(parent[2:6]))
        for daughter in daughters:
            p = rotate(daughter, rotation)
            p[2:6] = boost(p[2:6], beta)
            result.append(p)
    final = np.asarray([p[2:6] for p in result if p[1] == 1])
    if not np.all(np.isfinite(final)) or np.any(final[:, 3] <= 0.):
        raise KinematicsError("Reconstruction produced an invalid final-state momentum")
    if not np.allclose(final.sum(axis=0), target, rtol=0., atol=1.e-8 * target[3]):
        raise KinematicsError("Reconstruction failed four-momentum conservation")
    return result


def dot4vec(p, n):
    return p[5] * n[5] - np.dot(p[2:5], n[2:5])

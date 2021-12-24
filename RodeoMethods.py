import numpy as np
from qiskit import *


# This file has all of the methods necessary to run the two-qubit rodeo algorithm without the IMBQ job stuff. The  ...
# ... final method, identify_peaks(), is intentionally missing the method to get the counts for each run. This needs ...
# ... to be replaced with whatever method is used on each system. Note that IBMQ methods are still used to make each ...
# ... circuit
# turns "None" values to 0s. Useful for dealing with IBMQ output
def deNone(value):
    return int(0 if value is None else value)


# generate a set of times pulled from a normal distribution that 1. aren't too big and 2. aren't to close to each other
def make_good_times(stDev, num, minThreshold, maxThreshold):
    timesGood = False
    while not timesGood:
        times = np.abs(np.random.normal(0, stDev, num)).tolist()  # make a list of some normally-distributed times
        timesGood = True  # assume the times meet the requirements

        for i in times:  # check if any of the times are too big
            if i > maxThreshold:
                timesGood = False  # if any are, the times don't meet the requirements
                break

        if timesGood:  # if none of the times are too big, check if any are too close together
            times = np.sort(times)  # sort the times
            for i in range(num - 1):
                if times[i + 1] - times[i] < minThreshold:  # check if any neighbors are too close
                    timesGood = False  # if any are, the times don't meet the requirements
                    break

        if timesGood:
            return times


# make the circuit to be controlled using the controlled - reversal gates
# from Smith et al
def make_cont_sys_circ(inTime, xMod, zMod):
    temp = QuantumCircuit(3)  # initialize a quantum circuit with 3 qubits. Only 2 are used here, but it makes ...
    # appending easier

    temp.h(0)  # Hadamard gates change system from xMod XX + zMod ZZ to to xMod XZ + zMod ZX. This has results in ...
    # .. all 4 energy eigenvalues being present instead of just two

    # following is from Smith et al
    temp.cx(1, 0)

    temp.rx(2 * xMod * inTime, 1)
    temp.rz(2 * zMod * inTime, 0)

    temp.cx(1, 0)

    temp.h(0)  # second Hadamard to enable all eigenvalues

    return temp


# make a circuit with 1 cycle of the rodeo algorithm
# see original paper by Choi et al for
def make_cycle(time, ETarget, xMod, zMod):
    cycle = QuantumCircuit(3)  # all three qubits are used this time

    cycle.h(2)  # qubit 2 (the third one; 0 indexed) is the ancilla

    cycle.x(2)
    cycle.cy(2, 1)  # first controlled-reversal gate. Generates a phase difference from forwards/backwards time ...
    #  ... evolution instead of forward vs none

    cycle.compose(make_cont_sys_circ(time, xMod, zMod), [0, 1, 2], inplace=True)  # put in the system time evolution

    cycle.cy(2, 1)  # second controlled-reversal gate
    cycle.x(2)

    cycle.p(time * ETarget * 2, 2)

    cycle.h(2)

    return cycle


# creates a full circuit of the rodeo algorithm with some number of cycles and an array of times for each cycle
def run_rodeo(times, numCycles, ETarget, xMod, zMod):
    rodeo = QuantumCircuit(QuantumRegister(3), ClassicalRegister(numCycles))  # 3 qubits (2 for system, 1 ancilla) ...
    # ... and 1 cbit per cycle

    for i, time in enumerate(times):
        rodeo.compose(make_cycle(time, ETarget, xMod, zMod), [0, 1, 2], inplace=True)  # append each cycle
        rodeo.measure(2, i)  # add a mid-circuit measurement for the cycle

    return rodeo


# make a list with a circuit for each energy
# redundancy allows for multiple circuits per energy with unique random times for each one.
def make_scan(xMod, zMod, numCycles, energies, deviation, redundancy=1):
    circs = list()
    for i in energies:
        for k in range(redundancy):
            # ensure that times don't take too long/have bad overlaps
            # parameters 3 and 4 are technically arbitrary, but the ones are a good compromise between runtime and a ...
            # ...lack of secondary peaks
            times = make_good_times(deviation, numCycles, deviation / 5, deviation * 3)
            circs.append(run_rodeo(times, numCycles, i, xMod, zMod))
    return circs


# deal with IBMQ data output
# not called, but potentially useful for refrence
def clean_results(jobResults, number, numCycles, redundancy=1):
    state = ''
    duplicatedResults = list()
    # create the output state to be measured
    for i in range(numCycles):
        state = state + '0'
    # add redundant results to list of all results
    for runNum in range(number * redundancy):
        duplicatedResults.append(deNone(jobResults.get_counts(runNum).get(state)))
    # average redundant results
    # gives an array with total successes from each energy scan (averaged if there are multiple circuits per energy)
    averagedResults = np.mean(np.array(duplicatedResults).reshape(-1, redundancy), axis=1)
    return averagedResults


# finds peaks from the first pass given a list of energies and their successes
def find_first_peaks(firstRunCounts, firstRunEnergies, threshold):
    potentialPeaks = list()
    for i, numSuccesses in enumerate(firstRunCounts):
        # peak is defined as having a number of success over some threshold (typically 150 for three cycles)
        if numSuccesses >= threshold:
            potentialPeaks.append(firstRunEnergies[i])
    return potentialPeaks


# find peaks in the second scan results
def find_second_peaks(secondRunResults, secondRunEnergies, threshold):
    potentialPeaks = list()
    inPeak = False
    # find peaks using: second scan algorithm: go through each energy. if it is above 200, a peak is there. If the ...
    # ... next scan is greater, update the peak location. End peak when the next scan is below 200. Repeat for all scans
    for i, numSuccesses in enumerate(secondRunResults):
        # check if the current energy has more successes than the last energy
        if inPeak and numSuccesses > secondRunResults[i - 1]:
            # if it does, replace the energy for the peak with the energy with more successes
            # only do this in a peak
            potentialPeaks[-1] = secondRunEnergies[i]

        # check to see if there are 2 energies in a row with above-threshold successes
        # this method is good for wide peaks, but fails at small ones. With small peaks, it's possible for an energy ...
        # ... eigenvalue's neighboring energies to be below the threshold
        if not inPeak and numSuccesses > threshold and i + 1 < len(secondRunResults) and secondRunResults[
            i + 1] > threshold:
            # if there is, start a peak with the current energy as the peak
            potentialPeaks.append(secondRunEnergies[i])
            inPeak = True

        # check to see if the current energy has less successes than the threshold
        if inPeak and numSuccesses < threshold:
            # if it does, no longer a peak
            inPeak = False

    return potentialPeaks


# Zhengrong's fitting algorithm
# still under development
def find_final_peaks(finalRunResults):
    peaks = list()
    return peaks


def identify_peaks(xMod, zMod, numCycles, scanNums, finalShotsPerEnergy):
    maxEigenvalue = 5  # replace this with an upper/lower bound from xMod and zMod
    firstRunEnergies = np.linspace(-maxEigenvalue, maxEigenvalue, scanNums[0])  # I use 11 energies for the first scan,
    # ... but this is arbitrary
    firstStepSize = abs(firstRunEnergies[1]) - abs(firstRunEnergies[0])  # used to make second and third scans not ...
    # ... bleed into areas where there is no eigenvalue. Also prevents accidental overlap of scans

    firstPassCircuits = make_scan(xMod, zMod, numCycles, firstRunEnergies, 2, 1)
    firstRunCounts = list()  # list with total successes from each energy (averaged if there are multiple circuits ...
    # ... per energy)
    firstRunPeaks = find_first_peaks(firstRunCounts, firstRunEnergies,
                                     150)  # this can be replaced by any peak-finding algorithm for the first scan

    secondPassEnergies = list()
    for energy in firstRunPeaks:
        # go through each peak in the first scan and make a second scan over it. Append this scan to a list
        # number of energies is typically 11
        secondPassEnergies.append(np.linspace(energy + firstStepSize / 2, energy - firstStepSize / 2, scanNums[1]))
    secondPassEnergies = [item for sublist in secondPassEnergies for item in
                          sublist]  # collapse the array of second ...
    # ... scans into a single list of all energies to be run

    secondPassCircuits = make_scan(xMod, zMod, numCycles, secondPassEnergies, 7, 1)
    secondRunCounts = list()  # list with total successes from each energy (averaged if there are multiple circuits ...
    # ... per energy)
    secondRunPeaks = find_second_peaks(secondRunCounts, secondPassEnergies, 150)  # as with the first scan, this can ...
    # ... be replaced with whatever peak-finding algorithm is desired

    thirdPassEnergies = list()
    for i in secondRunPeaks:
        # do the same thing with the second scan peaks as was done with the first scan peaks.
        # there is one energy at the center of each peak from the second scan. The size of firstStepSize/4 is used ...
        # ... as there is a good bit of variation present from the second scan without redundancy
        # Basically scanning over just the energy of the second scan's peaks would probably miss some eigenvalues
        # Number of scans is typically 11 as well
        thirdPassEnergies.append(
            np.linspace(i + firstStepSize / 4, i - firstStepSize / 4, scanNums[2]))  # same process as ...
        # second scan: append the third scans to a list ...
    thirdPassEnergies = [item for sublist in thirdPassEnergies for item in
                         sublist]  # ... then collapse the list of lists

    thirdPassCircuits = make_scan(xMod, zMod, numCycles, thirdPassEnergies, 12, finalShotsPerEnergy)
    thirdRunCounts = list()  # list with total successes from each energy (averaged if there are multiple circuits ...
    # ... per energy)
    thirdRunPeaks = find_final_peaks(thirdRunCounts)  # currently not implemented

    # return can be modified to return data as needed
    return [secondRunCounts, secondPassEnergies, secondRunPeaks, thirdRunCounts, thirdPassEnergies]

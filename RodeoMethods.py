import lmfit.models
import numpy as np
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit import *
import matplotlib.pyplot as plt
from lmfit.models import ConstantModel, GaussianModel, SineModel
import random
import itertools
from qiskit import QuantumCircuit

file = open("key2.txt", "r")  # needs to be replaced with your key
key = file.read()
IBMQ.save_account(key, overwrite='True')
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research')
jobManager = IBMQJobManager()


# This file has all the methods necessary to run the two-qubit rodeo algorithm without the IMBQ job stuff. The  ...
# ... final method, identify_peaks(), is intentionally missing the method to get the counts for each run. This needs ...
# ... to be replaced with whatever method is used on each system. Note that IBMQ methods are still used to make each ...
# ... circuit
# turns "None" values to 0s. Useful for dealing with IBMQ output
def get_partial_key_matches(dictionary, partialKey):
    return dict(filter(lambda item: partialKey in item[0], dictionary.items()))


def sum_dict(dictionary):
    return np.sum(list(dictionary.values()))


def deNone(value):
    return int(0 if not value == value or value is None else value)


def flatten(ndlist):
    return [item for sublist in ndlist for item in sublist]


# returns the index of the element nearest to x
def closest_index(values, x):
    values = np.asarray(values)
    return (np.abs(values - x)).argmin()


# Gershgorin’s Theorem; turns out all the rows sum to the same thing and all diagonal elements are 0
def estimate_eignenvalues(xMod, zMod):
    return abs(xMod) + abs(zMod)


# generate a set of times pulled from a normal distribution that 1. aren't too big and 2. aren't to close to each other
def make_good_times(stDev, num, minThreshold, maxThreshold, overrideMin=False):
    timesGood = False
    while not timesGood:
        times = np.abs(np.random.normal(0, stDev, num)).tolist()  # make a list of some normally-distributed times
        timesGood = True  # assume the times meet the requirements

        for i in times:  # check if any of the times are too big
            if i > maxThreshold:
                timesGood = False  # if any are, the times don't meet the requirements
                break

        if timesGood and not overrideMin:  # if none of the times are too big, check if any are too close together
            times = np.sort(times)  # sort the times
            for i in range(num - 1):
                if times[i + 1] - times[i] < minThreshold:  # check if any neighbors are too close
                    timesGood = False  # if any are, the times don't meet the requirements
                    break

        if timesGood:
            return times


# make the circuit to be controlled using the controlled - reversal gates
# from Smith et al.
def make_cont_sys_circ(inTime, xMod, zMod):
    temp = QuantumCircuit(3)  # initialize a quantum circuit with 3 qubits. Only 2 are used here, but it makes ...
    # appending easier

    temp.h(2)  # Hadamard gates change system from xMod XX + zMod ZZ to xMod XZ + zMod ZX. This has results in ...
    # … all 4 energy eigenvalues being present instead of just two

    # following is from Smith et al.
    temp.cx(1, 2)

    temp.rx(2 * xMod * inTime, 1)
    temp.rz(2 * zMod * inTime, 2)

    temp.cx(1, 2)

    temp.h(2)  # second Hadamard to enable all eigenvalues

    return temp


# make a circuit with 1 cycle of the rodeo algorithm; see original paper by Choi et al. for detailed explanation of alg
def make_cycle(time, ETarget, xMod, zMod):
    cycle = QuantumCircuit(3)  # all three qubits are used this time

    cycle.h(0)  # qubit 0 (the first one; 0 indexed) is the ancilla

    cycle.x(0)
    cycle.cy(0, 1)  # first controlled-reversal gate. Generates a phase difference from forwards/backwards time ...
    # ... evolution instead of forward vs none

    cycle.compose(make_cont_sys_circ(time, xMod, zMod), [0, 1, 2], inplace=True)  # put in the system time evolution

    cycle.cy(0, 1)  # second controlled-reversal gate
    cycle.x(0)

    cycle.p(time * ETarget * 2, 0)

    cycle.h(0)

    return cycle


# creates a full circuit of the rodeo algorithm with some number of cycles and an array of times for each cycle
def run_rodeo_basic(times, numCycles, ETarget, xMod, zMod):
    rodeo = QuantumCircuit(QuantumRegister(3), ClassicalRegister(numCycles))  # 3 qubits (2 for system, 1 ancilla) ...
    # ... and 1 cbit per cycle

    for i, time in enumerate(times):
        rodeo.compose(make_cycle(time, ETarget, xMod, zMod), [0, 1, 2], inplace=True)  # append each cycle
        rodeo.measure(0, i)  # add a mid-circuit measurement for the cycle

    return rodeo


# creates a full circuit of the rodeo algorithm with options for two-state stuff
def run_rodeo(times, numCycles, ETarget, xMod, zMod, twoStateTime=None, measurements=None):
    # debug printing (currently disabled)
    # if twoStateTime is None:
    #     print('running energy = ' + str(ETarget))
    # else:
    #     print('running energy = ' + str(ETarget) + '; second time evolve is: ' + str(twoStateTime))

    rodeo = QuantumCircuit(QuantumRegister(3), ClassicalRegister(numCycles + 1))  # 3 qubits (2 for system, 1 ancilla)
    # ... and 1 cbit per cycle + 1 for the two-state rodeo algorithm measurement

    rodeo.x(1)
    rodeo.x(2)
    if measurements is not None:
        rodeo = QuantumCircuit(QuantumRegister(3), ClassicalRegister(numCycles + 2))  # add another cbit for more ...
        # ... 2-state rodeo algorithm measurements
        rodeo.x(1)
        rodeo.x(2)

    for i, time in enumerate(times):
        rodeo.compose(make_cycle(time, ETarget, xMod, zMod), [0, 1, 2], inplace=True)  # append each cycle
        rodeo.measure(0, i)  # add a mid-circuit measurement for the cycle
        # rodeo.reset(0)

    if twoStateTime is not None:
        rodeo.compose(make_cont_sys_circ(twoStateTime, xMod, zMod), [0, 1, 2], inplace=True)  # add a time-evolution ...
        # ... for the two-state rodeo algorithm

    match measurements:
        case "xx":
            rodeo.h(1)
            rodeo.measure(1, numCycles + 1)
            # cbit 0n000...
            rodeo.h(2)
            rodeo.measure(2, numCycles)
            # cbit n0000...
        case "zz":
            rodeo.measure(1, numCycles + 1)
            rodeo.measure(2, numCycles)
        case "xz":
            rodeo.h(1)
            rodeo.measure(1, numCycles + 1)
            rodeo.measure(2, numCycles)
        case "zx":
            rodeo.measure(1, numCycles + 1)
            rodeo.h(2)
            rodeo.measure(2, numCycles)

    return rodeo


def make_two_state_cycle(time, E0, E1, xMod, zMod):
    cycle = QuantumCircuit(4)  # all three qubits are used this time

    # qubit 0 (the first one; 0 indexed) is the ancilla
    cycle.h(1)

    cycle.x(1)
    cycle.cy(1, 2)  # first controlled-reversal gate. Generates a phase difference from forwards/backwards time ...
    # ... evolution instead of forward vs none

    cycle.compose(make_cont_sys_circ(time, xMod, zMod), [1, 2, 3], inplace=True)  # put in the system time evolution

    cycle.cy(1, 2)  # second controlled-reversal gate
    cycle.x(1)

    cycle.rz(time * 2 * E0, 1)
    cycle.crz(time * 2 * (E1-E0), 0, 1)

    cycle.h(1)

    return cycle


def make_two_state_scan(times, numCycles, E0, E1, xMod, zMod, theta, rotAxis):
    rodeo = QuantumCircuit(QuantumRegister(4), ClassicalRegister(2, "measurements"), ClassicalRegister(1, "ancilla"),
                           ClassicalRegister(numCycles, "cycles"))
    rodeo.h(0)
    for i, time in enumerate(times):
        rodeo.compose(make_two_state_cycle(time, E0, E1, xMod, zMod), [0, 1, 2, 3], inplace=True)
        rodeo.measure(1, i + 3)

    if rotAxis == "x":
        rodeo.rx(theta, 0)
    else:
        rodeo.ry(theta, 0)

    rodeo.measure(0, 2)
    rodeo.measure(2, 1)
    rodeo.measure(3, 0)

    return rodeo


def run_cont_two_state(cycles, redundancy, xMod, zMod, E0, E1, maxTheta, numScans, backend=provider.get_backend('ibmq_qasm_simulator'), jobID=None):
    xCircs = list()
    yCircs = list()
    angles = np.linspace(0, maxTheta, numScans)

    if jobID is None:
        for angle in angles:
            for i in range(redundancy):
                times = np.random.normal(0, 5, cycles)
                # xCircs.append(make_two_state_scan(times, cycles, E0, E1, xMod, zMod, angle, "x"))
                yCircs.append(make_two_state_scan(times, cycles, E0, E1, xMod, zMod, angle, "y"))
        circs = list()
        # circs.append(xCircs)
        circs.append(yCircs)
        circs = flatten(circs)
        circs = transpile(circs, backend=backend)
        job = jobManager.run(circs, backend=backend, name="two_state", shots=1024)
    else:
        job = jobManager.retrieve_job_set(jobID, provider)
    print("two state job " + job.job_set_id())

    return [job.results(), circs]


def make_two_state_test_cycle(time, E0, E1, xMod, zMod):
    cycle = QuantumCircuit(4)  # all three qubits are used this time

    # cycle.h(0)  # qubit 0 (the first one; 0 indexed) is the ancilla
    cycle.h(1)

    cycle.x(1)
    cycle.cy(1, 2)  # first controlled-reversal gate. Generates a phase difference from forwards/backwards time ...
    # ... evolution instead of forward vs none

    cycle.compose(make_cont_sys_circ(time, xMod, zMod), [1, 2, 3], inplace=True)  # put in the system time evolution

    cycle.cy(1, 2)  # second controlled-reversal gate
    cycle.x(1)

    # cycle.x(1)
    # cycle.p(-time*E0, 1)
    # cycle.x(1)
    # cycle.p(time * E0, 1)
    cycle.rz(time * 2 * E0, 1)
    # cycle.cx(0, 1)
    # cycle.cp(time * (E0-E1), 0, 1)
    # cycle.cx(0, 1)
    # cycle.cp(time * (E1 - E0), 0, 1)
    # cycle.crz(time * 2 * (E1 - E0), 0, 1)

    # cycle.h(0)
    cycle.h(1)

    return cycle


def test_cont_two_state(times, numCycles, E0, E1, xMod, zMod, twoStateTime):
    rodeo = QuantumCircuit(QuantumRegister(8), ClassicalRegister(1, "ancilla"), ClassicalRegister(2, "measurements"), ClassicalRegister(numCycles, "cycles"))
    rodeo.h(0)
    # rodeo.x(0)
    for i, time in enumerate(times):
        rodeo.compose(make_two_state_test_cycle(time, E0, E1, xMod, zMod), [0, i+1, 6, 7], inplace=True)
        rodeo.measure(i+1, i + 3)

    rodeo.h(0)
    # rodeo.x(0)
    # rodeo.measure(0, 0)

    # rodeo.h(2)
    # rodeo.h(3)
    # rodeo.measure(2, 2)
    # rodeo.measure(3, 1)
    rodeo.save_statevector()

    return rodeo


def make_cont_two_state_test_circs(cycles, redundancy, xMod, zMod, E0, E1, twoStateTimes):
    circs = list()

    for i in range(redundancy):
        times = np.random.normal(0, 5, cycles)
        for secondTime in twoStateTimes:
            circs.append(test_cont_two_state(times, cycles, E0, E1, xMod, zMod, secondTime))

    return circs


def run_cont_two_state_test(cycles, redundancy, xMod, zMod, E0, E1, twoStateTimes, backend=provider.get_backend('ibmq_qasm_simulator'), jobID=None):
    circs = list()

    if jobID is None:
        for i in range(redundancy):
            times = np.random.normal(0, 5, cycles)
            for secondTime in twoStateTimes:
                circs.append(test_cont_two_state(times, cycles, E0, E1, xMod, zMod, secondTime))
        circs = transpile(circs, backend=backend)
        job = jobManager.run(circs, backend=backend, name="two_state", shots=1024)
    else:
        job = jobManager.retrieve_job_set(jobID, provider)
    print("two state job " + job.job_set_id())

    return job.results()


def process_two_state_register(results, numCycles, redundancy, maxTheta, numScans):
    success = '0' * numCycles
    angles = np.linspace(0, maxTheta, numScans)
    states = ['00', '01', '10', '11']  # top down bot down, top down bot up, top up bot down, top up bot up
    overallAverageExpectation = 0
    overallQubitExpectations = [0, 0]
    redundantAverageExpectations = list()
    redundantQubitExpectations = list()
    all0Expectations = list()
    all1Expectations = list()
    allQubitExpectations = list()
    for index, angle in enumerate(angles):
        for redundantRun in range(redundancy):
            # filter successes

            successfulRuns0 = get_partial_key_matches(results.get_counts(index * redundancy + redundantRun), success + ' ' + '0')
            print(successfulRuns0)
            totalSuccesses0 = sum_dict(successfulRuns0)
            print(totalSuccesses0)

            successfulRuns1 = get_partial_key_matches(results.get_counts(index * redundancy + redundantRun), success + ' ' + '1')
            print(successfulRuns1)
            totalSuccesses1 = sum_dict(successfulRuns1)
            print(totalSuccesses1)
            # get number of  shots in all states
            stateResults0 = dict()
            stateResults1 = dict()
            for state in states:
                stateResults0[state] = sum_dict(get_partial_key_matches(successfulRuns0, success + " 0 " + state))
                stateResults1[state] = sum_dict(get_partial_key_matches(successfulRuns1, success + " 1 " + state))
            print(stateResults0)
            print(stateResults1)

            # calculate expectation value
            qubitExpectations = [0, 0]
            totalExpectation = 0
            for state in states:
                for qubit, qubitState in enumerate(state):
                    print(int(qubitState))
                    expectation = pow(-1, int(qubitState)) * stateResults1[state] / (totalSuccesses1)
                    totalExpectation += expectation
                    overallAverageExpectation += expectation
                    qubitExpectations[qubit] += expectation
                    overallQubitExpectations[qubit] += expectation
            redundantAverageExpectations.append(totalExpectation)
            redundantQubitExpectations.append(qubitExpectations)
    return [overallAverageExpectation, overallQubitExpectations, redundantAverageExpectations, redundantQubitExpectations]


def process_cont_test(results, numCycles, redundancy):
    # successful rodeo algorithm
    success = '0' * numCycles
    # first index is upper, second is lower
    # 0 for top and bot, 0 for top 1 for bot, 1 for top 0 for bot, 1 for top 1 for bot
    numCycles = 1
    successStates = ["".join(seq) for seq in itertools.product("01", repeat=numCycles)]
    for i, state in enumerate(successStates):
        successStates[i] = state + success

    totalSuccesses = 0
    for redundantRun in range(redundancy):
        for state in successStates:
            totalSuccesses += deNone(results.get_counts(redundantRun).get(state))

    return totalSuccesses/redundancy


# final, working processing method for energy-control two-state rodeo algorithm
def process_energy_controlled_two_state(results, numCycles, numAngles, redundancy):
    success = '0' * numCycles
    ancillaSuccessStates = [success + " 0", success + " 1"]

    systemStates = ['00', '01', '10', '11']

    stateCountsAncilla0 = dict()
    stateCountsAncilla1 = dict()

    expectations0 = list()
    expectations1 = list()

    for runIndex in range(numAngles):
        for redundant in range(redundancy):
            currentRunIndex = runIndex * redundancy + redundant

            resultsAncilla0 = get_partial_key_matches(results.get_counts(currentRunIndex), ancillaSuccessStates[0])
            ancilla0Total = sum_dict(resultsAncilla0)

            resultsAncilla1 = get_partial_key_matches(results.get_counts(currentRunIndex), ancillaSuccessStates[1])
            ancilla1Total = sum_dict(resultsAncilla1)

            for state in systemStates:
                stateCountsAncilla0[state] = sum_dict(get_partial_key_matches(resultsAncilla0, ancillaSuccessStates[0] + ' ' + state))
                stateCountsAncilla1[state] = sum_dict(get_partial_key_matches(resultsAncilla1, ancillaSuccessStates[1] + ' ' + state))
            print("total")
            print(sum_dict(get_partial_key_matches(results.get_counts(currentRunIndex), success)))
            print("ancilla = 0")
            print(ancilla0Total)
            print(resultsAncilla0)
            print(stateCountsAncilla0)
            print("ancilla = 1")
            print(ancilla1Total)
            print(resultsAncilla1)
            print(stateCountsAncilla1)

            print("expectations: ")
            expectations0.append(2 * stateCountsAncilla0["11"] / ancilla0Total - 2 * stateCountsAncilla0["00"] / ancilla0Total)
            print(2 * stateCountsAncilla0["11"] / ancilla0Total - 2 * stateCountsAncilla0["00"] / ancilla0Total)
            expectations1.append(2 * stateCountsAncilla1["11"] / ancilla1Total - 2 * stateCountsAncilla1["00"] / ancilla1Total)
            print(2 * stateCountsAncilla1["11"] / ancilla1Total - 2 * stateCountsAncilla1["00"] / ancilla1Total)

    averagedExpectations0 = np.mean(np.array(expectations0).reshape(-1, redundancy), axis=1)
    averagedExpectations1 = np.mean(np.array(expectations1).reshape(-1, redundancy), axis=1)

    return [averagedExpectations0, averagedExpectations1]


def process_two_state_dual_test(results, numCycles, numTimes, redundancy):
    # successful rodeo algorithm
    success = '0' * numCycles
    successStates = ['0' + success, '1' + success]

    totalExpectations = list()
    averageExpectations = [0] * numTimes

    # first index is upper, second is lower
    # 0 for top and bot, 0 for top 1 for bot, 1 for top 0 for bot, 1 for top 1 for bot
    systemStates = ['00', '01', '10', '11']
    for redundantRun in range(redundancy):
        totals = []
        countsUpper = []
        countsLower = []
        expectations = []
        for i in range(numTimes):
            currentRun = redundantRun*numTimes + i

            tempTotal = 0
            # get total successful counts
            for successfulState in successStates:
                for state in systemStates:
                    combinedState = state + successfulState
                    tempTotal += deNone(results.get_counts(currentRun).get(combinedState))
            totals.append(tempTotal)

            # append all 0 (00000) (bot down, top down) and 1 at start (10000) (bot down, top up)
            tempLower = 0
            tempUpper = 0
            for successfulState in successStates:
                tempLower += deNone(results.get_counts(currentRun).get(systemStates[0] + successfulState)) \
                                   + deNone(results.get_counts(i).get(systemStates[2] + successfulState))
                # append all 0 (00000) (bot down, top down) and 1 second (01000) (bot up, top down)
                tempUpper += deNone(results.get_counts(currentRun).get(systemStates[0] + successfulState)) \
                                   + deNone(results.get_counts(i).get(systemStates[1] + successfulState))

            countsLower.append(tempLower)
            countsUpper.append(tempUpper)
            # calculate the expectation value
            tempExpect = countsLower[i]/totals[i] * (1) + (1-countsLower[i]/totals[i]) * (-1)
            tempExpect = tempExpect + countsUpper[i]/totals[i] * (1) + (1-countsUpper[i]/totals[i]) * (-1)
            expectations.append(tempExpect)
        # print(expectations)
        totalExpectations.append(expectations)

    for expectationSet in totalExpectations:
        for timeNumber in range(len(averageExpectations)):
            averageExpectations[timeNumber] += expectationSet[timeNumber]

    for timeNumber in range(len(averageExpectations)):
        averageExpectations[timeNumber] = averageExpectations[timeNumber] / redundancy

    return [averageExpectations, totalExpectations]


def run_two_state_rodeo(times, numCycles, E0, E1, xMod, zMod, twoStateTime, measurements):
    rodeo = QuantumCircuit(QuantumRegister(4), ClassicalRegister(numCycles + 2))
    for i, time in enumerate(times):
        rodeo.compose(make_two_state_cycle(time, E0, E1, xMod, zMod), [0, 1, 2, 3], inplace=True)  # append each cycle
        rodeo.measure(1, i)  # add a mid-circuit measurement for the cycle
        # rodeo.reset(0)

    rodeo.compose(make_cont_sys_circ(twoStateTime, xMod, zMod), [1, 2, 3], inplace=True)

    match measurements:
        case "xx":
            rodeo.h(2)
            rodeo.measure(2, numCycles + 1)
            # cbit 0n000...
            rodeo.h(3)
            rodeo.measure(3, numCycles)
            # cbit n0000...
        case "zz":
            rodeo.measure(2, numCycles + 1)
            rodeo.measure(3, numCycles)
        case "xz":
            rodeo.h(2)
            rodeo.measure(2, numCycles + 1)
            rodeo.measure(3, numCycles)
        case "zx":
            rodeo.measure(2, numCycles + 1)
            rodeo.h(3)
            rodeo.measure(3, numCycles)

    return rodeo


def fit_controlled_two_state(processedResults, maxAngle, numMeasures, isY=True, printResults=False):
    angles = np.linspace(0, maxAngle, numMeasures)

    model = ConstantModel()
    model.set_param_hint('c', value=0, vary=True)
    model += SineModel(prefix='selfDifference_')
    model.set_param_hint('selfDifference_shift', value=np.pi/2, vary=False)
    model.set_param_hint('selfDifference_frequency', value=1, vary=False)
    model += SineModel(prefix='interactionElement_')

    if isY:
        model.set_param_hint('interactionElement_shift', value=0, vary=True)
    else:
        model.set_param_hint('interactionElement_shift', value=np.pi/2, vary=True)

    model.set_param_hint('interactionElement_frequency', value=1, vary=False)
    # grab the result and display it
    result = model.fit(processedResults[0], x=angles, method='nelder')

    if printResults:
        print(result.fit_report())
        plt.plot(angles, processedResults[0], 'o', ms=6)
        plt.plot(angles, result.best_fit, '-', label='best fit')
        plt.plot(angles, result.init_fit, '--', label='fit with initial values')
        plt.show()

    return [result.params, result.best_fit, result.best_values]


# make a list with a circuit for each energy
# redundancy allows for multiple circuits per energy with unique random times for each one.
def make_scan_basic(xMod, zMod, numCycles, energies, deviation, redundancy=1):
    circs = list()
    for i in energies:
        for k in range(redundancy):
            # ensure that times don't take too long/have bad overlaps
            # parameters 3 and 4 are technically arbitrary, but the ones are a good compromise between runtime and a ...
            # ...lack of secondary peaks
            times = make_good_times(deviation, numCycles, deviation / 5, deviation * 3)
            circs.append(run_rodeo_basic(times, numCycles, i, xMod, zMod))
    return circs


# make a list with a circuit for each energy
# redundancy allows for multiple circuits per energy with unique random times for each one.
def make_scan(xMod, zMod, numCycles, energies, deviation, redundancy=1):
    circs = list()
    for i in energies:
        for k in range(redundancy):
            # ensure that times don't take too long/have bad overlaps
            # parameters 3 and 4 are technically arbitrary, but the ones are a good compromise between runtime and a ...
            # ...lack of secondary peaks
            times = make_good_times(deviation, numCycles, deviation/5, deviation*3)
            circs.append(run_rodeo(times, numCycles, i, xMod, zMod))
    return circs


# deal with IBMQ data output
# not called, but potentially useful for reference
def clean_results(jobResults, number, numCycles, redundancy=1):
    state = '0'
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


def unflatten(flatResults, numberMerged):
    singleResultLength = int(len(flatResults) / numberMerged)
    arrayedResults = list()
    for i in range(numberMerged):
        resultSet = []
        for j in range(singleResultLength):
            resultSet.append(flatResults[i * singleResultLength + j])
        arrayedResults.append(resultSet)
    return arrayedResults


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
    # find peaks using:
    # second scan algorithm: go through each energy. if it is above 200, a peak is there.
    # If the next scan is greater, update the peak location. End peak when the next scan is below 200.
    # Repeat for all scans
    for i, numSuccesses in enumerate(secondRunResults):
        # check if the current energy has more successes than the last energy
        if inPeak and numSuccesses > secondRunResults[i - 1]:
            # if it does, replace the energy for the peak with the energy with more successes
            # only do this in a peak
            potentialPeaks[-1] = secondRunEnergies[i]

        # check to see if there are 2 energies in a row with above-threshold successes
        # this method is good for wide peaks, but fails at small ones. With small peaks, it's possible for an energy ...
        # ... eigenvalue's neighboring energies to be below the threshold
        if not inPeak and numSuccesses > threshold and i + 1 < len(secondRunResults) and secondRunResults[i + 1] > threshold:
            # if there is, start a peak with the current energy as the peak
            potentialPeaks.append(secondRunEnergies[i])
            inPeak = True

        # check to see if the current energy has fewer successes than the threshold
        if inPeak and numSuccesses < threshold:
            # if it does, no longer a peak
            inPeak = False

    return potentialPeaks


# creates a constrained gaussian curve
def make_gaussian_model(num, centerGuess, sigma):
    label = "peak{0}_".format(num)
    # initialize the model
    model = GaussianModel(prefix=label)
    # create constrained parameters with initial values
    model.set_param_hint(label + 'amplitude', value=25, min=0, max=300)
    model.set_param_hint(label + 'center', value=centerGuess)
    model.set_param_hint(label + 'sigma', value=sigma, min=0, max=.25)
    return model


# finds the peaks on the second cycle using a multi-gaussian fit
def second_peaks_gaussian(energies, counts, guesses, noiseLevel, errorThreshold, highThreshold, sigma, printResults=False):
    # start with a constant noise level
    model = ConstantModel()
    model.set_param_hint('c', value=noiseLevel, vary=False)

    # add a gaussian for each peak
    numPeaks = len(guesses)
    for i in range(numPeaks):
        model += make_gaussian_model(i, guesses[i], 1 / (2 * sigma))

    # get the result of the model
    result = model.fit(counts, x=energies, method='nelder')

    # display the results
    if printResults:
        print(result.fit_report())
        plt.plot(energies, counts, 'ro', ms=6)
        plt.plot(energies, result.best_fit, label='best fit')
        plt.plot(energies, result.init_fit, 'r--', label='fit with initial values')
        plt.show()

    rtn = list()
    for i in range(len(guesses)):
        centerLabel = "peak{0}_center".format(i)
        sigmaLabel = "peak{0}_sigma".format(i)
        temp = [result.params[centerLabel].value, result.params[centerLabel].stderr, result.params[sigmaLabel].value]
        # filter out peaks that don't meet an error threshold
        if deNone(temp[1])/temp[0] < errorThreshold and abs(deNone(temp[0])) < highThreshold and temp[2] < 1.1 / sigma:
            rtn.append(temp)

    return rtn


# implementation of Zhengrong's fitting algorithm. Not actually used
def noisy_gaussian_fit(energies, frequencies, guess, deviation):
    energies = np.array(energies)
    frequencies = np.array(frequencies)

    peak = GaussianModel(prefix='peak_')
    offset = ConstantModel(prefix='noise_')
    model = peak + offset

    parameters = model.make_params(gauss_center=guess, gauss_sigma=1/deviation)
    result = model.fit(frequencies, parameters, x=energies)

    return [result.params["peak_center"].value, result.params["peak_center"].stderr]


# finds the peaks on the third (final) scan using a multi-gaussian model
def find_initial_final_peaks(energies, counts, guesses, noiseLevel, sigma, printResults=False):
    # very similar to second_peaks_gaussian
    # start with a constant background noise
    model = ConstantModel()
    model.set_param_hint('c', value=noiseLevel, vary=False)
    for i in range(len(guesses)):
        # add gaussians for each peak
        model += make_gaussian_model(i, guesses[i][0], 1 / (2 * sigma))

    # grab the result and display it
    result = model.fit(counts, x=energies, method='nelder')

    if printResults:
        print(result.fit_report())
        plt.plot(energies, counts, 'ro', ms=6)
        plt.plot(energies, result.best_fit, label='best fit')
        plt.plot(energies, result.init_fit, 'r--', label='fit with initial values')
        plt.show()

    rtn = list()
    for i in range(len(guesses)):
        centerLabel = "peak{0}_center".format(i)
        sigmaLabel = "peak{0}_sigma".format(i)
        temp = [result.params[centerLabel].value, result.params[centerLabel].stderr, result.params[sigmaLabel].value]
        if deNone(temp[1])/temp[0] <= 0.1 and temp[2] < 1.1 / sigma:
            # filter out peaks that don't meet an error threshold
            rtn.append(temp)

    return rtn


# find each peak with an individual gaussian fit
def find_final_final_peaks(energies, counts, guesses, noiseLevel, sigma, scanNum, numEigenvalues, printResults=False):
    nearbyEnergies = list()
    nearbyCounts = list()

    for scan in range(numEigenvalues):
        tempEnergies = list()
        tempCounts = list()
        # find the nearest index to each guess
        for energyNum in range(scanNum):
            # add all the nearby energies and counts to a list
            tempEnergies.append(energies[scanNum * scan + energyNum])
            tempCounts.append(counts[scanNum * scan + energyNum])
        # add the list of nearby energies (and counts) for one peak to lists with this data for all of them
        nearbyEnergies.append(tempEnergies)
        nearbyCounts.append(tempCounts)

    rtn = list()
    # for each set of energies and counts, run a constant + (single) gaussian fit and get the center
    for (guessNum, energyList, freqList) in zip(range(len(guesses)), nearbyEnergies, nearbyCounts):
        rtn.append(find_single_peak(energyList, freqList, guesses[guessNum][0], noiseLevel, sigma, printResults=printResults))

    return rtn


def find_single_peak(energies, counts, guess, noiseLevel, sigma, printResults=False):
    model = ConstantModel()
    model.set_param_hint('c', value=noiseLevel, vary=False)
    model += make_gaussian_model(0, guess, 1 / (2 * sigma))

    # grab the result and display it
    result = model.fit(counts, x=energies, method='nelder')

    if printResults:
        print(result.fit_report())
        plt.plot(energies, counts, 'o', ms=6)
        plt.plot(energies, result.best_fit, '-', label='best fit')
        plt.plot(energies, result.init_fit, '--', label='fit with initial values')
        plt.show()

    centerLabel = "peak0_center"
    sigmaLabel = "peak0_sigma"
    return [result.params[centerLabel].value, result.params[centerLabel].stderr, result.params[sigmaLabel].value, result.best_fit, result.best_values]


# find peaks for a given set of parameters
def identify_peaks(xMod, zMod, numCycles, scanNums, shotsPerEnergy, maxEigenvalue=None, scanWidths=None, jobIDs=None, repetitions=None, deviations=None, shots=None, backend=provider.get_backend('ibmq_belem')):

    # only scan a certain range instead of the entire possible space
    maxEigenvalue = estimate_eignenvalues(xMod, zMod) if maxEigenvalue is None else maxEigenvalue
    # control the widths of the scans as a fraction of the first scan's width
    scanWidths = [0.5, 0.25] if scanWidths is None else scanWidths
    # allow for efficient repeated scans with the same parameters
    repetitions = 1 if repetitions is None else repetitions
    # control standard deviations for scans
    deviations = [2, 7, 12] if deviations is None else deviations
    # number of shots per circuit
    shots = 1024 if shots is None else shots

    noiseCounts = shots/(2**numCycles)

    firstRunEnergies = np.linspace(-maxEigenvalue, maxEigenvalue, scanNums[0])  # create a set of energies to scan first
    firstStepSize = abs(firstRunEnergies[1]) - abs(firstRunEnergies[0])  # used to make second and third scans not ...
    # ... bleed into areas where there is no eigenvalue. Also prevents accidental overlap of scans

    # makes a set of circuits to run (checks first scan for ranges where eigenvalues are likely)
    firstPassCircuits = list()
    for i in range(repetitions):
        firstPassCircuits.append(make_scan(xMod, zMod, numCycles, firstRunEnergies, deviations[0], shotsPerEnergy[0]))

    firstPassCircuits = flatten(firstPassCircuits)

    # IBMQ stuff start
    # create a job with all the circuits on the first pass if there isn't an id to retrieve from. Jobs with more than 300 circuits are split up
    if jobIDs is None or jobIDs[0] is None:
        firstPassJob = jobManager.run(transpile(firstPassCircuits, backend=backend), backend=backend, name="first_pass", shots=shots)
    else:
        firstPassJob = jobManager.retrieve_job_set(jobIDs[0], provider)
    firstPassJobID = firstPassJob.job_set_id()
    # print the id for later retrieval
    print("first run job id: " + firstPassJobID)
    # clean up the results into a more usable format
    firstRunCounts = clean_results(firstPassJob.results(), scanNums[0] * repetitions, numCycles, shotsPerEnergy[0])
    firstRunCounts = unflatten(firstRunCounts, repetitions)
    print(firstRunCounts)
    # IBMQ stuff end. Final result is a list with total successes from each energy scan (averaged if there are multiple circuits per energy)
    firstRunPeaks = list()
    for resultSet in firstRunCounts:
        firstRunPeaks.append(find_first_peaks(resultSet, firstRunEnergies, noiseCounts * 1.66))  # this can be replaced by any peak-finding algorithm for the first scan
    print(firstRunPeaks)

    secondPassEnergies = list()
    for energySet in firstRunPeaks:
        singleSecondPass = []
        # go through each peak in the first scan and make a second scan over it. Append this scan to a list
        # number of energies is typically 11
        for energy in energySet:
            singleSecondPass.append(np.linspace(energy + firstStepSize * scanWidths[0], energy - firstStepSize * scanWidths[0], scanNums[1]))
        singleSecondPass = flatten(singleSecondPass)  # collapse the array of second scans into a single list of all energies to be run
        secondPassEnergies.append(singleSecondPass)
    secondPassEnergies = flatten(secondPassEnergies)
    # make the circuits for the second scan
    secondPassCircuits = make_scan(xMod, zMod, numCycles, secondPassEnergies, deviations[1], shotsPerEnergy[1])

    # IBMQ stuff start
    # again, create and run a job with all the circuits  if there isn't an id to retrieve from
    if jobIDs is None or jobIDs[1] is None:
        secondPassJob = jobManager.run(transpile(secondPassCircuits, backend=backend), backend=backend, name="second_pass", shots=shots)
    else:
        secondPassJob = jobManager.retrieve_job_set(jobIDs[1], provider)
    # print the id to access the data later
    secondPassJobID = secondPassJob.job_set_id()
    print("second run job id: " + secondPassJobID)
    # take the data for the second scan and put in a usable format
    secondRunCounts = clean_results(secondPassJob.results(), len(secondPassEnergies), numCycles, shotsPerEnergy[1])
    secondRunCounts = unflatten(secondRunCounts, repetitions)
    secondPassEnergies = unflatten(secondPassEnergies, repetitions)
    print(secondPassEnergies)
    # IBMQ stuff end. Final result is the same as the first scan: a list with total successes from each energy scan (averaged if there are multiple circuits per energy)
    secondRunPeaks = list()
    for secondCountSet, secondEnergySet, firstPeakSet in zip(secondRunCounts, secondPassEnergies, firstRunPeaks):
        # makes a dictionary of the second scan counts keyed to their energies
        secondRunDict = {secondEnergySet[i]: secondCountSet[i] for i in range(len(secondEnergySet))}
        tempSecondRunDict = dict()

        # fill the space between the scans with noise
        for i in np.linspace(-maxEigenvalue, maxEigenvalue, scanNums[0]):
            for j in np.linspace(i - firstStepSize * scanWidths[0], i + firstStepSize * scanWidths[0], scanNums[1]):
                tempSecondRunDict[j] = secondRunDict.get(j, noiseCounts)

        # get the energies and counts for the second scan (including the added noise entries)
        baselineSecondEnergies = list(tempSecondRunDict.keys())
        baselineSecondCounts = list(tempSecondRunDict.values())
        # find the peaks of the second scan
        secondRunPeaks.append(second_peaks_gaussian(baselineSecondEnergies, baselineSecondCounts, firstPeakSet, noiseCounts, 0.1, maxEigenvalue * 1.25, deviations[1]))
        # as with the first scan, this can be replaced with whatever peak-finding algorithm is desired
    print(secondRunPeaks)

    thirdPassEnergies = list()
    for energySet in secondRunPeaks:
        singleThirdPass = []
        for i in energySet:
            # do the same thing with the second scan peaks as was done with the first scan peaks.
            # there is one energy at the center of each peak from the second scan. The size of firstStepSize/4 is used ...
            # ... as there is a good bit of variation present from the second scan without redundancy
            # Basically scanning over just the energy of the second scan's peaks would probably miss some eigenvalues
            # Number of scans is typically 11 as well
            singleThirdPass.append(np.linspace(i[0] + firstStepSize * scanWidths[1], i[0] - firstStepSize * scanWidths[1], scanNums[2]))  # same process as ...
            # second scan: append the third scans to a list ...
        singleThirdPass = flatten(singleThirdPass)
        thirdPassEnergies.append(singleThirdPass)  # ... then collapse the list of lists
    thirdPassEnergies = flatten(thirdPassEnergies)
    thirdPassCircuits = make_scan(xMod, zMod, numCycles, thirdPassEnergies, deviations[2], shotsPerEnergy[2])

    # IBMQ stuff start
    # make and run a job for the third scan if there isn't an id to retrieve from
    if jobIDs is None or jobIDs[2] is None:
        thirdPassJob = jobManager.run(transpile(thirdPassCircuits, backend=backend), backend=backend, name="third_pass", shots=shots)
    else:
        thirdPassJob = jobManager.retrieve_job_set(jobIDs[2], provider)
    # print the id and get usable results
    thirdPassJobID = thirdPassJob.job_set_id()
    print("third run job id: " + thirdPassJobID)
    thirdRunCounts = clean_results(thirdPassJob.results(), len(thirdPassEnergies), numCycles, shotsPerEnergy[2])
    thirdRunCounts = unflatten(thirdRunCounts, repetitions)
    thirdPassEnergies = unflatten(thirdPassEnergies, repetitions)
    # IBMQ stuff end; result is the same list of successes as usual

    initialThirdRunPeaks = list()
    finalThirdRunPeaks = list()
    # make a dictionary of the second scan counts keyed to their energies
    for thirdCountSet, thirdEnergySet, secondPeakSet in zip(thirdRunCounts, thirdPassEnergies, secondRunPeaks):
        thirdRunDict = {thirdEnergySet[i]: thirdCountSet[i] for i in range(len(thirdEnergySet))}
        tempthirdRunDict = dict()

        # fill the space between the scans with noise
        for i in np.linspace(-maxEigenvalue, maxEigenvalue, scanNums[0]):
            for j in np.delete(np.linspace(i - firstStepSize * scanWidths[0], i + firstStepSize * scanWidths[0], scanNums[1]), -1):
                tempthirdRunDict[j] = thirdRunDict.get(j, noiseCounts)

        # get the energies and counts for the second scan (including the added noise entries)
        baselineThirdEnergies = list(thirdRunDict.keys())
        baselineThirdCounts = list(thirdRunDict.values())
        # find the scans using multi-gaussian model
        tempInitialThirdRunPeaks = find_initial_final_peaks(baselineThirdEnergies, baselineThirdCounts, secondPeakSet, noiseCounts, deviations[2])
        initialThirdRunPeaks.append(tempInitialThirdRunPeaks)
        # print(initialThirdRunPeaks)
        # then get (theoretically) better estimates with a single gaussian
        finalThirdRunPeaks.append(find_final_final_peaks(thirdEnergySet, thirdCountSet, tempInitialThirdRunPeaks, noiseCounts, deviations[2], scanNums[2], 4))
    print(finalThirdRunPeaks)

    # return can be modified to return data as needed
    return {"finalPeaks": finalThirdRunPeaks, "secondPeaks": secondRunPeaks, "thirdDual": [thirdRunCounts, thirdPassEnergies],
            "secondDual": [secondRunCounts, secondPassEnergies], "firstDual": [firstRunCounts, firstRunEnergies], "initialThirdPeaks": initialThirdRunPeaks,
            "jobIDs": [firstPassJobID, secondPassJobID, thirdPassJobID]}


# plot a scan
def plot_scans(data):
    energy = []
    successes = []  # number of counts
    for i in data:
        for j in i:
            energy.append(j[0])
            successes.append(deNone(j[1]))

    hist1 = plt.bar(energy, successes, width=(0.8*(energy[1]-energy[0])))
    plt.show()


# method to test for the overlap after n cycles
def get_simple_overlap(numCycles, eigenvalues, xMod, zMod, numTries, twoStateEnergies=None, backend=provider.get_backend('ibmq_belem')):

    circs = list()
    counts = list()
    # for each eigenvalue, add numTries circuits at numCycles cycles to a list
    for energy in eigenvalues:
        energyCircs = []
        for i in range(numTries):
            if twoStateEnergies is not None:
                tempTimes = []
                for cycle in range(numCycles):
                    # make times that target two states
                    tempTime = round(np.random.normal(0, 12))
                    if tempTime == 0:
                        tempTime = pow((-1), random.randint(0, 1))
                    tempTimes.append(tempTime * 2 * 3.1415926 / (twoStateEnergies[0] - twoStateEnergies[1]))
            else:
                tempTimes = make_good_times(12, numCycles, 12 / 5, 12 * 5, overrideMin=True)
            energyCircs.append(run_rodeo(tempTimes, numCycles, energy, xMod, zMod))
        circs.append(energyCircs)

    for energyCircs, eigenvalue in zip(circs, eigenvalues):
        # run the set of circuits for each energy
        overlapJob = jobManager.run(transpile(energyCircs, backend=backend), backend=backend, name="overlapJob", shots=1024)
        print(str(eigenvalue) + " overlapJob job id: " + overlapJob.job_set_id())
        # get the average of each of the tries and add it to a list
        average = clean_results(overlapJob.results(), 1, numCycles, numTries)[0]
        counts.append(average)
        print(average)

    return counts


# methods for running the two-state rodeo algorithm. Should probably be refactored

# not used in favor of process_two_state_dual
def process_two_state(results, state):
    totals = list()
    counts = list()
    expectations = list()

    states = [state, '1' + state[:-1]]

    for i in range(100):
        temp = deNone(results.get_counts(i).get(states[0]))
        temp = temp + deNone(results.get_counts(i).get(states[1]))

        totals.append(temp)

    for i in range(100):
        counts.append(deNone(results.get_counts(i).get(state)))

    for i in range(100):
        expectations.append(counts[i]/totals[i] * (-1) + (1-counts[i]/totals[i]) * 1)

    print(totals)
    return expectations


# process two-state rodeo algorithm data with two-cbit observables
def process_two_state_dual(results, numCycles, numTimes, redundancy):
    # successful rodeo algorithm
    success = '0' * numCycles

    totalExpectations = list()
    averageExpectations = [0] * numTimes

    # first index is upper, second is lower
    # 0 for top and bot, 0 for top 1 for bot, 1 for top 0 for bot, 1 for top 1 for bot
    states = ['00' + success, '01' + success, '10' + success, '11' + success]
    for redundantRun in range(redundancy):
        totals = []
        countsUpper = []
        countsLower = []
        expectations = []
        for i in range(numTimes):
            currentRun = redundantRun*numTimes + i

            # get total successful counts
            temp = deNone(results.get_counts(currentRun).get(states[0]))
            temp = temp + deNone(results.get_counts(currentRun).get(states[1]))
            temp = temp + deNone(results.get_counts(currentRun).get(states[2]))
            temp = temp + deNone(results.get_counts(currentRun).get(states[3]))
            totals.append(temp)

            # append all 0 (00000) (bot down, top down) and 1 at start (10000) (bot down, top up)
            countsLower.append(deNone(results.get_counts(currentRun).get(states[0])) + deNone(results.get_counts(i).get(states[2])))
            # append all 0 (00000) (bot down, top down) and 1 second (01000) (bot up, top down)
            countsUpper.append(deNone(results.get_counts(currentRun).get(states[0])) + deNone(results.get_counts(i).get(states[1])))

            # calculate the expectation value
            tempExpect = countsLower[i]/totals[i] * (1) + (1-countsLower[i]/totals[i]) * (-1)
            tempExpect = tempExpect + countsUpper[i]/totals[i] * (1) + (1-countsUpper[i]/totals[i]) * (-1)
            expectations.append(tempExpect)
        # print(expectations)
        totalExpectations.append(expectations)

    for expectationSet in totalExpectations:
        for timeNumber in range(len(averageExpectations)):
            averageExpectations[timeNumber] += expectationSet[timeNumber]

    for timeNumber in range(len(averageExpectations)):
        averageExpectations[timeNumber] = averageExpectations[timeNumber] / redundancy

    return [averageExpectations, totalExpectations]


def run_two_state(cycles, redundancy, secondTimes, xMod, zMod, lowTarget, highTarget, measurements, backend=provider.get_backend('ibmq_belem'), jobID=None):
    circs = list()

    # allows for retrieval of jobs
    if jobID is None:
        for i in range(redundancy):
            times = []
            # for each cycle, make times that target both states
            for cycle in range(cycles):
                tempTime = round(np.random.normal(0, 5))
                if tempTime == 0:
                    tempTime = pow((-1), random.randint(0, 1))
                times.append(tempTime * 2 * 3.1415926 / (lowTarget - highTarget))
            # for each secondary time evolution time, add a circuit with the above rodeo algorithm times
            for time in secondTimes:
                circs.append(run_rodeo(times, cycles, lowTarget, xMod, zMod, twoStateTime=time, measurements=measurements))

        # run the circuits
        circs = transpile(circs, backend=backend)
        job = jobManager.run(circs, backend=backend, name="two_state", shots=1024)
    else:
        job = jobManager.retrieve_job_set(jobID, provider)
    print("two state job" + job.job_set_id())

    return process_two_state_dual(job.results(), cycles, len(secondTimes), redundancy)


def run_controlled_two_state(cycles, redundancy, twoStateTimes, xMod, zMod, E0, E1, measurements, backend=provider.get_backend('ibmq_belem'), jobID=None):
    circs = list()

    if jobID is None:
        for i in range(redundancy):
            times = np.random.normal(0, 5, cycles)
            for time in twoStateTimes:
                circs.append(run_two_state_rodeo(times, cycles, E0, E1, xMod, zMod, time, measurements))
        circs = transpile(circs, backend=backend)
        job = jobManager.run(circs, backend=backend, name="two_state", shots=1024)
    else:
        job = jobManager.retrieve_job_set(jobID, provider)
    print("two state job " + job.job_set_id())

    return process_two_state_dual(job.results(), cycles, len(twoStateTimes), redundancy)

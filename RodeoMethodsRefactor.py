import numpy as np
import qiskit.providers
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit import *
import matplotlib.pyplot as plt
from lmfit.models import ConstantModel, GaussianModel
import random

file = open("key2.txt", "r") # needs to be replaced with your key
key = file.read()
IBMQ.save_account(key, overwrite='True')
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research')
jobManager = IBMQJobManager()


def deNone(value):
    return int(0 if not value == value or value is None else value)

def flatten(ndlist):
    return [item for sublist in ndlist for item in sublist]

class SysInfo:
    def __init__(self, size, mods):
        # mods is list of {'x': xMod, 'z': zMod, 'y': yMod} triplets
        self.size = size
        self.mods = mods

    def get_mod(self, n):
        return self.mods[n]

# returns the index of the element nearest to x
def closest_index(values, x):
     values = np.asarray(values)
     return (np.abs(values - x)).argmin()

#Gershgorin’s Theorem; turns out all the rows sum to the same thing and all diagonal elements are 0
def estimate_eignenvalues(sysParams: SysInfo):
    return abs(sysParams.get_mod(0)['x']) + abs(sysParams.get_mod(0)['z'])

# generate a set of times pulled from a normal distribution that 1. aren't too big and 2. aren't to close to each other
def make_good_times(stDev, num, minThreshold, maxThreshold, overrideMin = False):
    timesGood = False
    while not timesGood:
        times = np.abs(np.random.normal(0, stDev, num)).tolist() # make a list of some normmally-distributed times
        timesGood = True # assume the times meet the requirements

        for i in times: # check if any of the times are too big
            if i > maxThreshold:
                timesGood = False # if any are, the times don't meet the requirements
                break

        if timesGood and not overrideMin: # if none of the times are too big, check if any are too close together
            times = np.sort(times) # sort the times
            for i in range(num - 1):
                if times[i + 1] - times[i] < minThreshold: # check if any neighbors are too close
                    timesGood = False # if any are, the times don't meet the requirements
                    break

        if timesGood:
            return times

# make the circuit to be controlled using the controlled - reversal gates
# from Smith et al
def make_system(inTime, mods):
    temp = QuantumCircuit(3) # initialize a quantum circuit with 3 qubits. Only 2 are used here, but it makes ...
    # appending easier

    temp.h(2) # Hadamard gates change system from xMod XX + zMod ZZ to to xMod XZ + zMod ZX. This has results in ...
    # .. all 4 energy eigenvalues being present instead of just two

    # following is from Smith et al
    temp.cx(1, 2)

    temp.rx(2 * mods['x'] * inTime, 1)
    temp.rz(2 * mods['z'] * inTime, 2)

    temp.cx(1, 2)

    temp.h(2) # second Hadamard to enable all eigenvalues

    return temp


# make a circuit with 1 cycle of the rodeo algorithm; see original paper by Choi et al for detailed explanation of alg
def make_cycle(time, ETarget, mods):
    cycle = QuantumCircuit(3) # all three qubits are used this time

    cycle.h(0) # qubit 0 (the first one; 0 indexed) is the ancilla

    cycle.x(0)
    cycle.cy(0, 1) # first controlled-reversal gate. Generates a phase difference from forwards/backwards time evolution ...
    # ... instead of forward vs none

    cycle.compose(make_system(time, mods), [0, 1, 2], inplace=True) # put in the system time evolution

    cycle.cy(0, 1) # second controlled-reversal gate
    cycle.x(0)

    cycle.p(time * ETarget * 2, 0)

    cycle.h(0)

    return cycle

# creates a full circuit of the rodeo algorithm with some number of cycles and an array of times for each cycle
def run_rodeo_basic(times, numCycles, ETarget, sysParams: SysInfo):
    rodeo = QuantumCircuit(QuantumRegister(sysParams.size() + 1), ClassicalRegister(numCycles))  # 3 qubits (2 for system, 1 ancilla) ...
    # ... and 1 cbit per cycle

    for i, time in enumerate(times):
        rodeo.compose(make_cycle(time, ETarget, sysParams.get_mod(i)), [0, 1, 2], inplace=True)  # append each cycle
        rodeo.measure(0, i)  # add a mid-circuit measurement for the cycle

    return rodeo

class PassInfo:
    def __init__(self, passNumber, redundancy=None, sigma=None, numScans=None):
        self.passNumber = passNumber
        match passNumber:
            case 1:
                self.redundancy = redundancy if redundancy is not None else 5
                self.sigma = sigma if sigma is not None else 2
                self.numScans = numScans if numScans is not None else 11
            case 2:
                self.redundancy = redundancy if redundancy is not None else 2
                self.sigma = sigma if sigma is not None else 7
                self.numScans = numScans if numScans is not None else 11
            case 3:
                self.redundancy = redundancy if redundancy is not None else 1
                self.sigma = sigma if sigma is not None else 12
                self.numScans = numScans if numScans is not None else 22

    def get_pass_number(self):
        return self.passNumber
    def get_number_of_scans(self):
        return self.numScans
    def get_redundancy(self):
        return self.redundancy
    def get_stdev(self):
        return self.sigma
    def get_param_dict(self):
        return {"passNumber": self.passNumber, "numScans": self.numScans, "redundancy": self.redundancy, "sigma": self.sigma}

class Run:
    def __init__(self, sysParams: SysInfo, passParams=None, cycles=None, shots=None, backend=provider.get_backend('ibmq_belem')):
        self.sysParams = sysParams
        self.backend = backend

        self.maxEigenvalue = estimate_eignenvalues(self.sysParams)
        self.energyRange = [-self.maxEigenvalue, self.maxEigenvalue]

        self.cycles = cycles if cycles is not None else 3
        self.shots = shots if shots is not None else 1024
        self.passParams = passParams if passParams is not None else [PassInfo(1), PassInfo(2), PassInfo(3)]

    def get_cycles(self):
        return self.cycles
    def get_sys_params(self):
        return self.sysParams
    def get_backend(self):
        return self.backend
    def get_shots(self):
        return self.shots
    
    class Pass:
        def __init__(self, bounds, passParams: PassInfo, run):
            self.bounds = bounds
            self.params = passParams
            self.runInstance = run

            self.
            for each

            self.energies = np.linspace(self.bounds[0], self.bounds[1], self.params.get_number_of_scans())

            self.averageCounts = list()

        def get_params(self):
            return self.params
        def get_run(self):
            return self.runInstance
        class Sweep:
            def __init__(self, energies, parentPass, jobID = None):
                self.energies = energies
                self.passInstance = parentPass
                self.jobID = jobID if jobID is not None else None
                self.job = qiskit.providers.Job()

                self.counts = list()
                self.circuits = list()
            def make_scans(self):
                for energy in self.energies:
                    # ensure that times don't take too long/have bad overlaps
                    # parameters 3 and 4 are technically arbitrary, but the ones are a good compromise between runtime and a ...
                    # ...lack of secondary peaks
                    times = make_good_times(self.passInstance.get_params().get_stdev(), self.passInstance.get_run().get_cycles(), self.passInstance.get_params().get_stdev() / 5, self.passInstance.get_params().get_stdev() * 3)
                    self.circuits.append(run_rodeo_basic(times, self.passInstance.get_run().get_cycles(), energy, self.passInstance.get_run().get_sys_params()))

            def submit_job(self):
                if self.jobID is None:
                    self.job = jobManager.run(transpile(flatten(self.circuits), backend=self.passInstance.get_run().get_backend()), backend=self.passInstance.get_run().get_backend(), name="pass_{runNum}".format(runNum=self.passInstance.get_params().get_pass_number(), shots=self.passInstance.get_run().get_shots()))
                else:
                    self.job = jobManager.retrieve_job_set(self.jobID, provider=provider)
                self.jobID = self.job.job_set_id()

            def extract_results(self):
                self.job = jobManager.retrieve_job_set(self.jobID, provider=provider)
                state = '0' * self.passInstance.get_run().get_cycles()
                duplicateResults = list()

                for runNum in range(self.passInstance.get_params().get_number_of_scans() * self.passInstance.get_params().get_redundancy()):
                    duplicateResults.append(deNone(self.job.results().get_counts(runNum).get(state)))

                self.counts = np.mean(np.array(duplicateResults).reshape(-1, self.passInstance.get_params().get_redundancy()), axis=1)

            def get_counts(self):
                return self.counts

        def average_
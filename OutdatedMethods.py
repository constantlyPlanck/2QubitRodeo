from RodeoMethods import *


# OUTDATED - see identify_peaks and submethods for a better implementation
def search_spectrum(linspace, threshold, numCycles, laterScanNum, numSecond, xMod, zMod, backend = provider.get_backend('ibmq_qasm_simulator')):
    delta = abs(linspace[1]) - abs(linspace[0])
    length = linspace[0] - linspace[-1]

    runResults = list()
    energyList = list()

    state = ''
    for i in range(numCycles):
        state = state + '0'
    state = state + '0'

    peaks = list()
    firstRunCircs = list()
    for i in linspace:
        times = []
        for j in range(numCycles):
            times.append(np.random.normal(0, 2))
        firstRunCircs.append(run_rodeo(times, numCycles, i, xMod, zMod))
    print("first pass w/ energies")
    print(linspace)
    firstRunCircs = transpile(firstRunCircs, backend=backend)
    firstRunJob = jobManager.run(firstRunCircs, backend=backend, name = "first_pass", shots=1024)
    print("first run job id: " + firstRunJob.job_set_id())
    firstRunResults = firstRunJob.results()
    runResults.append(firstRunResults)
    energyList.append(linspace)

    for runNum in range(linspace.size):
        if firstRunResults.get_counts(runNum).get(state) is not None and firstRunResults.get_counts(runNum).get(state) >= threshold:
            peaks.append(linspace[runNum])
    print("second pass w/ peaks:")
    print(peaks)

    secondRunCircs = list()
    energyList.append([])
    for i in peaks:
        for j in np.linspace(i - delta/2, i + delta/2, laterScanNum):
            for num in range(numSecond):
                times = []
                for k in range(numCycles):
                    times.append(np.random.normal(0, 7))
                # print('energy: ' + str(j))
                secondRunCircs.append(run_rodeo(times, numCycles, j, xMod, zMod))
                energyList[1].append(j)

    secondRunCircs = transpile(secondRunCircs, backend=backend)
    secondRunJob = jobManager.run(secondRunCircs, backend=backend, name = "second_run", shots=1024)
    print("second run job id: " + secondRunJob.job_set_id())

    runResults.append(secondRunJob.results())
    print("done w/ second pass")

    #third scan algorithm: go through each energy. if it is above 200, a peak is there. If the next scan is greater, update the peak location. End peak when the next scan is below 200. Repeat for all scans

    return [runResults, energyList, state, [11, laterScanNum], [1, numSecond]]


#this code is genuinely terrible, but it works. It's also now outdated by identify_peaks
#runNumber is 0 indexed
def process_data(output, runNumber):
    pairArray = []
    for i in range(int(len(output[1][runNumber]) / (output[4][runNumber]))):
        totalSuccess = 0
        for j in range(output[4][runNumber]):
            if output[0][runNumber].get_counts(i * output[4][runNumber] + j).get(output[2]) is not None:
                totalSuccess = totalSuccess + output[0][runNumber].get_counts(i * output[4][runNumber] + j).get(output[2])
        pairArray.append([output[1][runNumber][i * output[4][runNumber]], totalSuccess / output[4][runNumber]])

    subScans = []
    for i in range(int(len(pairArray) / output[3][runNumber])):
        temp = []
        for k in range(output[3][runNumber]):
            temp.append(pairArray[k + i * output[3][runNumber]])
        temp.reverse()
        subScans.append(temp)
    return subScans
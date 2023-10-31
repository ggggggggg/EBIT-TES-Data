import mass


import numpy as np 
import matplotlib.pyplot as plt 
from mass.off import ChannelGroup, getOffFileListFromOneFile
import h5py
import os
import matplotlib
import mass.calibration._highly_charged_ion_lines

plt.close('all')
plt.ion()

dir = '.'
day = "20231015"
run = "0000"
output_dirname = 'output'

cal_states = ['C']
fine_cal_states = ["G","H"]
# fine_cal_states=["C"]

file = getOffFileListFromOneFile(os.path.join(dir,day,run,f"{day}_run{run}_chan1.off"), maxChans=400)

data = ChannelGroup(file)
data.setDefaultBinsize(0.8)
ds = data[2]
# ds.plotHist( np.arange(0, 60000, 20), "filtValue", coAddStates=False, states=cal_states)   #states=None by default uses all states
#plt.show()

ds.learnDriftCorrection(overwriteRecipe=True, states=cal_states)
ds.calibrationPlanInit("filtValueDC")
ds.calibrationPlanAddPoint(10067, "AlKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(11708, "SiKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(17267, "ClKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(21483, "KKAlpha", states=cal_states)
data.learnDriftCorrection(overwriteRecipe=True, states=fine_cal_states)
data.alignToReferenceChannel(ds, "filtValueDC", np.arange(0,50000,30),states=cal_states)
# ds.calibrateFollowingPlan("filtValueDC")
data.learnPhaseCorrection(uncorrectedName="filtValueDC",correctedName="filtValueDCPC", states=fine_cal_states)
# data.learnDriftCorrection(indicatorName="relTimeSec", uncorrectedName="filtValueDCPC", 
#                           correctedName="filtValueDCPCTC", states="E")
data.calibrateFollowingPlan("filtValueDCPC")
data.calcExternalTriggerTiming(after_last=True)
filename = data.outputHDF5Filename(outputDir=output_dirname,addToName="listmode")
with h5py.File(filename,"w") as h5:
    data.energyTimestampLabelToHDF5(h5)

def firstAndLastRelTimeSecOfStates(states):
    return ds.getAttr("relTimeSec",indsOrStates=states, cutRecipeName="cutNone")[[0,-1]]
a,b = firstAndLastRelTimeSecOfStates(fine_cal_states)
ds.plotAvsB("relTimeSec", "energy")
plt.grid(True)
plt.ylim(500,2000)
plt.xlim(a,b)
ds.plotAvsB2d("relTimeSec", "energy", (np.arange(a,b,300), np.arange(100,2000,1)), norm=matplotlib.colors.LogNorm())

def midpoints(x):
    d = x[1]-x[0]
    return x[0]+(np.arange(len(x)-1)+0.5)*d
def timeFit(self,relTimeSec_start, relTimeSec_end, bin_duration_s, line, attr, dlo, dhi, params_update):
    n_time_bins = int(np.ceil((relTimeSec_end-relTimeSec_start)/bin_duration_s))
    relTimeSec_edges = np.linspace(relTimeSec_start, relTimeSec_end)
    inds = np.searchsorted(ds.getAttr("relTimeSec", indsOrStates=None, 
                                      cutRecipeName="cutNone"), relTimeSec_edges)
    results = []
    for i in range(len(inds)-1):
        # add a temporary state for this time bin fo efficiency indexing
        ds.statesDict["_temp_state"] = slice(inds[i], inds[i+1], None)
        result = self.linefit(line, attr=attr, states="_temp_state", plot=False, params_update=params_update)
        results.append(result)
    ds.statesDict.popitem("_temp_state") # remove it from the states dict

    plt.figure()
    plt.errorbar(midpoints(relTimeSec_edges), 
             [result.params["peak_ph"].value for result in results], 
             yerr=[result.params["peak_ph"].stderr for result in results], fmt="o")
    plt.grid(True)
    plt.xlabel("relTimeSec")
    plt.ylabel(f"peak_ph (eV) attr={attr}")
    plt.title(self.shortName)
    plt.tight_layout()
    
    return results
dlo, dhi = 14,14
result = ds.linefit("AlKAlpha",dlo=dlo,dhi=dhi,states=fine_cal_states)
params = result.params.copy()
params["fwhm"].set(vary=False)
results = timeFit(ds, a,b, 1000, "AlKAlpha", dlo=dlo, dhi=dhi,attr="energy", params_update = params)

data.learnDriftCorrection(indicatorName="relTimeSec", 
                          uncorrectedName="energy", states=fine_cal_states)
results = timeFit(ds, a,b, 1000, "AlKAlpha", dlo=dlo, dhi=dhi,attr="energyDC", params_update = params)
data.learnDriftCorrection(indicatorName="relTimeSec", 
                          uncorrectedName="filtValueDCPC",
                            correctedName = "filtValueDCPCTC",
                              states=fine_cal_states)

for ds in data.values():
    ds.calibrationPlanInit("filtValueDCPCTC")
    prev_cal = cal=ds.recipes["energy"].f
    ds.calibrationPlanAddPoint(prev_cal.energy2ph(mass.STANDARD_FEATURES["AlKAlpha"]), 
                               "AlKAlpha", states=fine_cal_states)
    ds.calibrationPlanAddPoint(prev_cal.energy2ph(mass.STANDARD_FEATURES["SiKAlpha"]), 
                               "SiKAlpha", states=fine_cal_states)
# ds.calibrationPlanAddPoint(17267, "ClKAlpha", states=cal_states)
# ds.calibrationPlanAddPoint(21483, "KKAlpha", states=cal_states)
data.calibrateFollowingPlan("filtValueDCPCTC", calibratedName="energyDC2",dlo=dlo, dhi=dhi, approximate=False)
cal=ds.recipes["energyDC2"].f
results = timeFit(ds, a,b, 1000, "AlKAlpha", dlo=dlo, dhi=dhi,attr="energyDC2", params_update = params)
ds.diagnoseCalibration("energyDC2")
data.qualityCheckLinefit("AlKAlpha", worstAllowedFWHM=4.5, dlo=dlo, dhi=dhi, 
                         positionToleranceAbsolute=0.2, attr="energyDC2",
                         states=fine_cal_states)
result_mgka = data.linefit("MgKAlpha", dlo=dlo/2, dhi=dhi/2,states=fine_cal_states)
def human_lang_result(result):
    name = result.model.spect.shortname
    peak_ph_param = result.params["peak_ph"]
    v, e = peak_ph_param.value, peak_ph_param.stderr
    expected = result.model.spect.peak_energy
    print(f"""{name} fit peak energy {v:.3f}+/-{e:.3f} eV is {v-expected:.3f} eV above expected value""")
human_lang_result(result_mgka)
mass.line_models.VALIDATE_BIN_SIZE = False
result_o2p = data.linefit("O H-Like 2p", attr="energyDC2", dlo=dlo, dhi=dhi,states=fine_cal_states,
                          minimum_bins_per_fwhm=0.5)
human_lang_result(result_o2p)
result_o3p = data.linefit("O H-Like 3p", attr="energyDC2",dlo=dlo, dhi=dhi/2,states=fine_cal_states,
                          minimum_bins_per_fwhm=0.5)
human_lang_result(result_o3p)
result_alk = data.linefit("AlKAlpha", attr="energyDC2",dlo=dlo, dhi=dhi,states=fine_cal_states)
human_lang_result(result_alk)
human_lang_result(data.linefit("SiKAlpha", attr="energyDC2",dlo=dlo, dhi=dhi,states=fine_cal_states))
human_lang_result(data.linefit("ClKAlpha", attr="energyDC2",dlo=dlo, dhi=dhi,states=fine_cal_states))

plt.figure()
for ds in data.values()[:8]:
    ds.plotHist(np.arange(500,3000,1), attr="energyDC2", states=fine_cal_states, axis=plt.gca(),
                labelLines=["AlKAlpha", "MgKAlpha", "SiKAlpha"])
plt.legend(labels=[f"channel {ds.channum}" for ds in data.values()[:8]])
plt.title(f"states={fine_cal_states} for channels in legends {data.shortName}")
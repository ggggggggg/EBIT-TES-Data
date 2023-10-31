import mass


import numpy as np 
import matplotlib.pyplot as plt 
from mass.off import ChannelGroup, getOffFileListFromOneFile
import h5py
import os
import matplotlib
import mass.calibration._highly_charged_ion_lines
import lmfit

plt.close('all')
plt.ion()

dir = '.'
day = "20231015"
run = "0000"
output_dirname = 'output'

cal_states = ['C']
fine_cal_states = ["G","H"]
# fine_cal_states=["C"]

file = getOffFileListFromOneFile(os.path.join(dir,day,run,f"{day}_run{run}_chan1.off"), maxChans=800)

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
    ds.calibrationPlanAddPoint(prev_cal.energy2ph(mass.STANDARD_FEATURES["ClKAlpha"]), 
                               "ClKAlpha", states=fine_cal_states)
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
for ds in data.values()[:2]:
    ds.plotHist(np.arange(500,3000,1), attr="energyDC2", states=fine_cal_states, axis=plt.gca(),
                labelLines=["AlKAlpha", "MgKAlpha", "SiKAlpha"])
plt.legend(labels=[f"channel {ds.channum}" for ds in data.values()[:8]])
plt.title(f"states={fine_cal_states} for channels in legends {data.shortName}")

# plt.close("all")
def align_fit(channum2, elo, ehi, doplot):
    # elo, ehi = 500, 2000
    # data.cutAdd("cutEnergyROI",lambda energy: np.logical_and(energy>elo, energy<ehi), overwrite=True)
    # test polynominal alignment
    ds1 = data[1]
    ds2 = data[channum2]
    ds1.cutAdd("cutEnergyROI",lambda energy: np.logical_and(energy>elo, energy<ehi), overwrite=True)
    ds2.cutAdd("cutEnergyROI",lambda energy: np.logical_and(energy>elo, energy<ehi), overwrite=True)
    x1_unscaled = ds1.getAttr("filtValueDCPCTC", indsOrStates=fine_cal_states, cutRecipeName="cutEnergyROI")
    x2_unscaled = ds2.getAttr("filtValueDCPCTC", indsOrStates=fine_cal_states, cutRecipeName="cutEnergyROI")
    # get some fixed points to start our polynominal with
    cal1 = ds1.recipes["energyDC2"].f
    cal2 = ds2.recipes["energyDC2"].f
    fvs1_unscaled = np.array([0] + list(cal1._ph))
    fvs2_unscaled = np.array([0] + list(cal2._ph))
    scale = 10000.0
    x1 = x1_unscaled/scale
    x2 = x2_unscaled/scale
    fvs1 = fvs1_unscaled/scale
    fvs2 = fvs2_unscaled/scale
    poly2to1 = np.polynomial.polynomial.Polynomial.fit(fvs2, fvs1, deg=len(fvs1)-1).convert()
    print(f"{poly2to1=}")
    if doplot:
        plt.figure()
        plt.plot(fvs1,"bo", label="fvs1")
        plt.plot(poly2to1(fvs2),"r.", label="poly2to1(fvs2)")
        plt.plot(np.arange(len(x1))+len(fvs1),x1,"bo", label = "x1")
        plt.plot(np.arange(len(x2))+len(fvs1),poly2to1(x2), "r.", label="poly2to1(x2)")
        plt.title(f"chan {channum2} mapping to chan 1")
        plt.xlabel("sample number")
        plt.ylabel("filt value mapped to chan1")
        plt.legend()

    a=mass.mathstat.entropy.laplace_KL_divergence(scale*x1[::10], scale*x2[::10])
    b=mass.mathstat.entropy.laplace_KL_divergence(scale*x1[::10], scale*poly2to1(x2[::10]))
    print(f"{a=:.03f} {b=:.03f}")
    def params2poly(params):
        valuedict = params.valuesdict()
        coefs = []
        for letter in "ABCDEFG":
            if letter in valuedict.keys():
                coefs.append(valuedict[letter])
        return np.polynomial.polynomial.Polynomial(coefs)    

    def residual(params, x1, x2):
        poly2to1 = params2poly(params)
        return mass.mathstat.entropy.laplace_KL_divergence(scale*x1, scale*poly2to1(x2))

    params0 = lmfit.Parameters()
    for i, val in enumerate(poly2to1.coef):
        if np.abs(val) > 1e-10:
            params0.add("ABCDEFG"[i],val, vary=True, min=-1, max=1)
        else:
            params0.add("ABCDEFG"[i],0, vary=False)
    params0.add("A",0,vary=False)
    # b=poly2to1.coef[1]
    b = params0["B"].value
    params0.add("B",b, max=b*1.05, min=b*0.95)
    # params.add("C",1e-6)
    # params.add("D",1e-10)
    print(f"{params0=}")
    minimizer0 = lmfit.Minimizer(residual, params0, (x1[::10], x2[::10]))
    result0 = minimizer0.minimize("nelder")
    poly2to1_fit0 = params2poly(result0.params)
    
    params = result0.params.copy()
    # params["A"].set(1e-9, vary=True, min=-0.1, max=0.1)
    params.add("E", 0.001, min=-1, max=1)
    minimizer = lmfit.Minimizer(residual, params, (x1, x2))
    result = minimizer.minimize("nelder")

    # result = result0
    poly2to1_fit = params2poly(result.params)

    ebin_edges = np.arange(elo,ehi)
    fv_edges = cal1.energy2ph(ebin_edges)/scale
    c1, _ = np.histogram(x1, fv_edges)
    c2, _ = np.histogram(poly2to1_fit(x2), fv_edges)
    c2_orig, _ = np.histogram(poly2to1(x2), fv_edges)
    c2_orig2, _ = np.histogram(cal1.energy2ph(cal2.ph2energy(x2*scale))/scale, fv_edges)
    if doplot:
        plt.figure()
        plt.plot(midpoints(ebin_edges), c1, label="c1")
        plt.plot(midpoints(ebin_edges), c2, label="c2")
        plt.plot(midpoints(ebin_edges), c2_orig, label="c2_orig")
        plt.plot(midpoints(ebin_edges), c2_orig2, label="c2_orig2")
        plt.legend()
        plt.xlabel("energy (eV) approx")
        plt.ylabel("counts be 1 eV bin")
        plt.title(f"chan {channum2} mapping to chan 1")

    # ds1.recipes.add("filtValueDCPCTC_chan1", lambda filtValueDCPCTC: filtValueDCPCTC, overwrite=True)
    ds2.recipes.add("filtValueDCPCTC_chan1", lambda filtValueDCPCTC: scale*poly2to1_fit(filtValueDCPCTC/scale), overwrite=True)
    ds2.poly2to1_fit = poly2to1_fit
    ds2.poly2to1_fit_result = result
    ds2.scale = scale


for ds in data.values():
    print(f"align_fit on chan {ds.channum}")
    align_fit(ds.channum, 500, 2000, doplot=False)
    print(ds.poly2to1_fit)
    print(f"{ds.poly2to1_fit_result.residual=}")
    print()

for ds in data.values():
    if ds.poly2to1_fit_result.residual[0] > 0.1:
        ds.markBad("sucks at poly2to1_fit_result.residual")

data.plotHist(np.arange(0,40000,10), attr="filtValueDCPCTC_chan1", states=fine_cal_states)
filtValueDCPCTC_chan1_cal = data[1].recipes["energyDC2"].f
for ds in data.values():
    ds.recipes.add("energy3", 
                   lambda filtValueDCPCTC_chan1: filtValueDCPCTC_chan1_cal(filtValueDCPCTC_chan1),
                   overwrite=True)
result_alk3 = data.linefit("AlKAlpha", attr="energy3", states=fine_cal_states, dlo=dlo, dhi=dhi)
result_mgk3 = data.linefit("MgKAlpha", attr="energy3", states=fine_cal_states, dlo=dlo, dhi=dhi)
result_sik3 = data.linefit("SiKAlpha", attr="energy3", states=fine_cal_states, dlo=dlo, dhi=dhi)
model_o2p3 = mass.spectra["O H-Like 2p"].model() + mass.get_model(666.3, has_linear_background=False, prefix="e_")
params_o2p3 = model_o2p3.make_params()
params_o2p3["e_fwhm"].set(expr="fwhm")
params_o2p3["e_dph_de"].set(1, vary=False)
params_o2p3["dph_de"].set(1, vary=False)
e_peak = mass.spectra["O H-Like 2p"].peak_energy
elo, ehi = e_peak-dlo, e_peak+dhi*2
bin_edges_o2p3 = np.arange(elo, ehi, ds._default_bin_size)
_, counts_o2p3 = data.hist(attr="energy3",binEdges=bin_edges_o2p3,
                          states=fine_cal_states)
result_o2p3 = model_o2p3.fit(counts_o2p3, bin_centers = midpoints(bin_edges_o2p3), params=params_o2p3)
result_o2p3.plotm()
data.plotHist(attr="energy3"), binEdges = np.arange(500,2000), states=fine_cal_states)

standardSettings = Dict{String,Any}()

_dt = 1.0 # [ms]/[step]
_tau = 10.0 # [ms]
_presentationLength = 100 # [ms]

kernelfunction(t) = exp(-t)
#kernelfunction(t) = exp(-t^2/2)

standardSettings["showProgressBar"] = false

standardSettings["comment"] = ""

### Generating the dataset
# 100 ms presentation length in steps
standardSettings["presentationLength"] = Int(_presentationLength / _dt) # [step]
# 'fades' between image-presentations, denotes fraction of fade to constant
standardSettings["fadeLength"] = 0.3
standardSettings["inputMotionProbability"] = 0.1


### Network architecture
standardSettings["n_x"] = NaN
standardSettings["n_y"] = NaN
standardSettings["n_v1"] = NaN
standardSettings["n_m2"] = NaN

### Neuronal dynamics
# simulation time step - if this is changed later in the simulation,
# use 'set_dt'
standardSettings["dt"] = _dt
# time constant of PSP decay, 10ms
standardSettings["kernelTau"] = _tau / _dt # [step]
standardSettings["kernelLength"] = Int(5 * Int(ceil(standardSettings["kernelTau"])))
standardSettings["kernel"] = [kernelfunction(t/standardSettings["kernelTau"])
                              for t in 0:standardSettings["kernelLength"]-1]
# target rate
standardSettings["rhov1"] = 0.02 # 1/[ms]
standardSettings["rhom2"] = 0.02 # 1/[ms]

### For generating the random initial params
standardSettings["weightVariance"] = 0.0
standardSettings["weightMean"] = 0.0
# The initial sigma of the gaussian
standardSettings["initialSigmaV1"] = 1.0
standardSettings["initialSigmaM2"] = 1.0


### Learning settings
standardSettings["learningRateFeedForwardV1"] = 0.00006
standardSettings["learningRateHomeostaticBiasV1"] = 0.001
standardSettings["learningRateFeedForwardM2"] = 0.00006
standardSettings["learningRateHomeostaticBiasM2"] = 0.001
# update only every nth timestep
standardSettings["updateInterval"] = 1

### Online-updates of parameters
# First key is name of parameter, second is time of change
# So to change "learningRateSigma" at t=100 to 0.1:
# s["paramChangeDict"]["learningRateSigma"] = Dict(100 => 0.1)
standardSettings["paramChangeDict"] = Dict{String, Dict{Int64, Any}}()


### Logging settings
# Length of averaging window for templog
standardSettings["tempLogInterval"] = 1000
# Sample performance every nth timestep
standardSettings["tempLogSampleInterval"] = 1
# Save only every nth timestep in the snapshot to save space
standardSettings["snapshotLogInterval"] = 1
standardSettings["testWeightsNecessity"] = false


""" Save way of changing settings, so typos are noticed."""
function set!(s::Dict{String, Any}, key::String, value::Any)
    @assert (key in keys(s)) "Can't set setting. Key $key not valid."
    if (key == "dt")
        set_dt(s, value)
    else
        s[key] = value
    end
end

function set_dt(s::Dict{String, Any}, dt::Any)
    s["dt"] = dt
    s["presentationLength"] = Int(round(_presentationLength / dt))
    s["kernelTau"] = _tau / dt # [step]
    s["kernelLength"] = Int(round(5 * s["kernelTau"]))
    s["kernel"] = [kernelfunction(t/s["kernelTau"])
                    for t in 0:s["kernelLength"]-1]
end

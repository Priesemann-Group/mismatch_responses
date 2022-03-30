include("../src/analytic/settings.jl")
include("../src/analytic/main.jl")

using Profile
import Random

function main(plotflag, s, seed=1234)
    print("Setting up...\n")
    nPatterns = 50000
    nTestPatterns = 300
    imageDim = 8
    scale = 0.5

    dt = 0.2
    set!(s, "comment", "")
    set!(s, "showProgressBar", true)

    set!(s, "dt", dt)
    set!(s, "tempLogSampleInterval", Int(5.0/dt))
    set!(s, "updateInterval", Int(ceil(0.2/dt)))
    set!(s, "snapshotLogInterval", Int(1.0/dt))

    l = s["presentationLength"]
    nSteps = nPatterns * l
    set!(s, "tempLogInterval", 1000 * l)

    set!(s, "n_x", 6) # number x neurons
    set!(s, "n_y", 2) # number y neurons
    set!(s, "n_v1", 12) # number v1 neurons
    set!(s, "n_m2", 2) # number m2 neurons
    set!(s, "weightVariance", 0.2)
    
    set!(s, "inputMotionProbability", 0.2)

    set!(s, "initialSigmaV1", sqrt(0.05))
    set!(s, "initialSigmaM2", sqrt(0.10))

    set!(s, "learningRateFeedForwardV1", 3e-4)
    set!(s, "learningRateFeedForwardM2", 3e-4)
    
    s["paramChangeDict"]["learningRateHomeostaticBiasV1"] =
        Dict(1         => 1e-2,
             40000 * l => 3e-4)
    s["paramChangeDict"]["learningRateHomeostaticBiasM2"] =
        Dict(1         => 5e-4,
             40000 * l => 3e-4)

    set!(s, "rhov1", 8.0 / 1000.0)
    set!(s, "rhom2", 30.0 / 1000.0) 

    Random.seed!(seed)
    inputs, egomotion = create_visual_flow_inputs(nPatterns, s)
    test_inputs, test_egomotion = create_visual_flow_inputs(nTestPatterns, s, test=true)

    test_times = collect(0:div(nSteps, 3):nSteps)

    return main("mismatch", plotflag, s, nSteps, inputs, egomotion, test_inputs, test_egomotion, test_times)
end

plotflag = true
main(plotflag, copy(standardSettings))

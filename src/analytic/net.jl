using ProgressMeter
using LinearAlgebra: diagind
using BSON

mutable struct Net
    n_x :: Int32
    x_outputs :: Array{Float64,1}
    n_y :: Int32
    y_outputs :: Array{Float64,1} # labels: egomotion

    n_v1 :: Int32
    v1_neurons :: Array{Neuron,1}
    v1_outputs :: Array{Float64,1}
    v1_spikes :: Array{Bool,1}
    v1_rates :: Array{Float64,1}
    v1_input :: Array{Float64,1}
    
    n_m2 :: Int32
    m2_neurons :: Array{Neuron,1}
    m2_outputs :: Array{Float64,1}
    m2_spikes :: Array{Bool,1}
    m2_rates :: Array{Float64,1}
    m2_input :: Array{Float64,1}

    xv1_weights :: Array{Float64, 2}
    xm2_weights :: Array{Float64, 2}
    ym2_weights :: Array{Float64, 2}

    v1_biases :: Array{Float64,1}
    m2_biases :: Array{Float64,1}

    sigma_V1 :: Float64
    sigma_V1_2 :: Float64 # sigma^(-2) !
    sigma_M2 :: Float64
    sigma_M2_2 :: Float64 # sigma^(-2) !

    # reconstruction of the current input
    reconstruction :: Array{Float64,1}

    memory :: Dict{String, Any}

    log :: Log
end

function create_net(s::Dict{String,Any}, log::Log)::Net
    n_x = s["n_x"]::Int
    n_y = s["n_y"]::Int
    n_v1 = s["n_v1"]::Int
    n_m2 = s["n_m2"]::Int
    rhov1 = s["rhov1"]::Float64
    rhom2 = s["rhom2"]::Float64
    dt = s["dt"]::Float64
    initialSigmaV1 = s["initialSigmaV1"]::Float64
    initialSigmaM2 = s["initialSigmaM2"]::Float64

    memory = Dict()

    x_outputs = zeros(n_x)
    y_outputs = zeros(1)

    v1_neurons = [create_neuron(s) for i in 1:n_v1]
    v1_outputs = zeros(n_v1)
    v1_input = zeros(n_v1)
    v1_spikes  = falses(n_v1)
    v1_rates = zeros(n_v1)
    
    m2_neurons = [create_neuron(s) for i in 1:n_m2]
    m2_outputs = zeros(n_m2)
    m2_input = zeros(n_m2)
    m2_spikes  = falses(n_m2)
    m2_rates = zeros(n_m2)

    sigma_V1 = initialSigmaV1
    sigma_V1_2 = sigma_V1 ^ (-2)
    sigma_M2 = initialSigmaM2
    sigma_M2_2 = sigma_M2 ^ (-2)

    xv1_weights = create_xv1_connections(s)
    xm2_weights = create_xm2_connections(s)
    ym2_weights = create_ym2_connections(s)
    
    v1_biases = logit(rhov1 * dt) * ones(n_v1)
    m2_biases = logit(rhom2 * dt) * ones(n_m2)

    reconstruction = zeros(n_x)

    net = Net(n_x, x_outputs, n_y, y_outputs,
              n_v1, v1_neurons, v1_outputs, v1_spikes, v1_rates, v1_input,
              n_m2, m2_neurons, m2_outputs, m2_spikes, m2_rates, m2_input,
              xv1_weights, xm2_weights, ym2_weights, v1_biases, m2_biases, sigma_V1, sigma_V1_2,
              sigma_M2, sigma_M2_2, reconstruction, memory, log)
    return net
end

""" Runs the network on a dataset and a testset.

 - x_inputs: (nPatterns, n)-Matrix with input strengths.
 - test_input: (:, n)-Matrix with test-inputs.
 - test_times: Specifies at what times the snapshots are taken.
"""
function run_net(net::Net, x_inputs::Array{Float64,2}, egomotion::Array{Float64,2}, test_input::Array{Float64,2}, test_egomotion::Array{Float64,2},
    test_times::Array{Int,1}, save_loc::String, s::Dict{String,Any})

    l = s["presentationLength"]::Int
    interval = s["tempLogInterval"]::Int
    updateInterval = s["updateInterval"]::Int
    changeDict = s["paramChangeDict"]::Dict{String,Dict{Int64,Any}}
    showProgressBar = s["showProgressBar"]::Bool

    nSteps = size(x_inputs,1) * l
    runningLog = setup_temp_log(net, nSteps, interval)
    
    dtBar = showProgressBar ? 0.1 : Inf
    @showprogress dtBar for t in 1:nSteps
        net.log.t = t

        # get input to net by fading between images
        x = @inbounds fade_images(x_inputs,t,s)
        y = @inbounds fade_images(egomotion,t,s)

        # update network
        batchUpdate = t % updateInterval == 0 # use "batch" update to save time
        step_net(net, x, y, s, update=true, batchUpdate=batchUpdate)

        # log everything
        log_everything(x, test_input, test_egomotion, runningLog, test_times, save_loc, net, s)

        # online-update of parameters
        for (param, timeDict) in changeDict
            if haskey(timeDict, t)
                s[param] = timeDict[t]
            end
        end
    end
end

""" Takes the network and steps one timestep forward.

 - update: If true, the weights will be updated.
"""
function step_net(net::Net, x_input::Array{Float64,1}, egomotion::Array{Float64,1}, s::Dict{String,Any};
    update::Bool = false, batchUpdate::Bool = false)

    net.x_outputs = x_input
    net.y_outputs = egomotion

    # fire neurons
    net.v1_input = calc_v1_input(net, s)
    net.v1_spikes = map(step_neuron, net.v1_neurons, net.v1_input)
    
    net.m2_input = calc_m2_input(net, s)
    net.m2_spikes = map(step_neuron, net.m2_neurons, net.m2_input)
    
    # in rare event of simultaneous spikes we remove the least likely
    # spikes to improve stability (also works without this, but preferably 
    # the precision should be decreased in this case)
    reduce_spikes_to_one_spike(net, s)
    
    # update parameters
    if update
        if batchUpdate
            update_net(net, s)
        end
    end  

    # update outputs
    net.v1_outputs = Array{Float64,1}(map(get_output, net.v1_neurons))
    net.m2_outputs = Array{Float64,1}(map(get_output, net.m2_neurons))

    # reconstruct the current "prediction" of the input
    net.reconstruction = reconstruct_input(net, s)
end


function reduce_spikes_to_one_spike(net::Net, s::Dict{String,Any})
    if sum(net.v1_spikes) > 1
        _, ind_spiker = findmax([net.v1_spikes[j] * net.v1_neurons[j].prob for j in 1:net.n_v1])
        for j in 1:net.n_v1
            if net.v1_spikes[j] && j != ind_spiker
                pop!(net.v1_neurons[j].recentSpikes)
                net.v1_neurons[j].prob = 1.0 - net.v1_neurons[j].prob
                net.v1_spikes[j] = false
            end
        end
    end
    if sum(net.m2_spikes) > 1
        _, ind_spiker = findmax([net.m2_spikes[j] * net.m2_neurons[j].prob for j in 1:net.n_m2])
        for j in 1:net.n_m2
            if net.m2_spikes[j] && j != ind_spiker
                pop!(net.m2_neurons[j].recentSpikes)
                net.m2_neurons[j].prob = 1.0 - net.m2_neurons[j].prob
                net.m2_spikes[j] = false
            end
        end
    end
    if sum(net.v1_spikes) + sum(net.m2_spikes) > 1
        p1 = [net.v1_spikes[j] * net.v1_neurons[j].prob for j in 1:net.n_v1]
        p2 = [net.m2_spikes[j] * net.m2_neurons[j].prob for j in 1:net.n_m2]
        ps = vcat(p1, p2)
        _, ind_spiker = findmax(ps)
        for j in 1:net.n_v1
            if net.v1_spikes[j] && j != ind_spiker
                pop!(net.v1_neurons[j].recentSpikes)
                net.v1_neurons[j].prob = 1.0 - net.v1_neurons[j].prob
                net.v1_spikes[j] = false
            end
        end
    end
end


function write_status(save_loc::String, net::Net, log::Log)
    try
        save_net(net, save_loc)
    catch e
        print("Error saving net:\n")
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        println(msg)
    end
    try
        save_log(log, save_loc)
    catch e
        print("Error saving log:\n")
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        println(msg)
    end
end

function log_everything(x::Array{Float64,1}, test_input::Array{Float64,2}, test_egomotion::Array{Float64,2}, runningLog::Dict{String, Any}, 
    test_times::Array{Int,1}, save_loc::String, net::Net, s::Dict{String, Any})
    
    interval = s["tempLogInterval"]::Int
    sampleInterval = s["tempLogSampleInterval"]::Int
    numSamples = div(interval, sampleInterval)
    updateInterval = s["updateInterval"]::Int
    changeDict = s["paramChangeDict"]::Dict{String,Dict{Int64,Any}}

    # rates have to be updated for homeostasis
    net.v1_rates += net.v1_spikes / updateInterval
    net.m2_rates += net.m2_spikes / updateInterval
    runningLog["firing_rates_v1"] += net.v1_spikes / interval
    runningLog["firing_rates_m2"] += net.m2_spikes / interval

    # update & save the log at certain time intervals
    t = net.log.t

    if t % sampleInterval == 0
        update_running_log(net, runningLog, x, interval, numSamples, s)
    end
    if t % interval == 0 
        log_temp_log(net, runningLog, test_input, test_egomotion, interval, s)
    end
    if t in test_times
        take_snapshot(net, net.log, test_input, test_egomotion, s)
        write_status(save_loc, net, net.log)
    end
end

function save_net(net::Net, name::String)
    rm("$name/net.bson", force=true)
    bson("$name/net.bson", net = net)
end


""" A Snapshot saves the dynamics and state
of the network at a given moment in time."""
mutable struct Snapshot
    t :: Int

    v1_spikes :: Array{Bool}
    m2_spikes :: Array{Bool}
    x_outputs :: Array{Float64,2}
    y_outputs :: Array{Float64,2}
    v1_inputs :: Array{Float64,2}
    m2_inputs :: Array{Float64,2}
    v1_outputs :: Array{Float64,2}
    m2_outputs :: Array{Float64,2}

    xv1_weights :: Array{Float64, 2}
    xm2_weights :: Array{Float64, 2}
    ym2_weights :: Array{Float64, 2}
    v1_biases :: Array{Float64,1}
    m2_biases :: Array{Float64,1}

    sigma_2 :: Float64

    test_inputs :: Array{Float64, 2}
    reconstructions :: Array{Float64, 2}
    reconstructions_y :: Array{Float64, 2}
    reconstruction_means :: Array{Float64, 2}
    reconstruction_vars :: Array{Float64, 2}
end

function create_snapshot(nSteps::Int, interval::Int, t::Int, inputs::Array{Float64,2}, 
    net::Net, s::Dict{String,Any})::Snapshot
    
    v1_spikes = falses(div(nSteps,interval), s["n_v1"])
    m2_spikes = falses(div(nSteps,interval), s["n_m2"])
    x_outputs = zeros(div(nSteps,interval), s["n_x"])
    y_outputs = zeros(div(nSteps,interval), s["n_y"])
    v1_inputs = zeros(div(nSteps,interval), s["n_v1"])
    m2_inputs = zeros(div(nSteps,interval), s["n_m2"])
    v1_outputs = zeros(div(nSteps,interval), s["n_v1"])
    m2_outputs = zeros(div(nSteps,interval), s["n_m2"])

    v1_biases = net.v1_biases
    m2_biases = net.m2_biases
    l = s["presentationLength"]::Int
    reconstructions = zeros(div(nSteps,interval), s["n_x"])
    reconstructions_y = zeros(div(nSteps,interval), s["n_y"])
    reconstruction_means = zeros(div(nSteps,l), s["n_x"])
    reconstruction_vars = zeros(div(nSteps,l), s["n_x"])
    return Snapshot(t, v1_spikes, m2_spikes, x_outputs, y_outputs, v1_inputs, m2_inputs, v1_outputs, m2_outputs,
                    copy(net.xv1_weights), copy(net.xm2_weights), copy(net.ym2_weights),
                    v1_biases, m2_biases, copy(net.sigma_V1_2),
                    inputs, reconstructions, reconstructions_y,
                    reconstruction_means, reconstruction_vars)
end

""" Creates the snapshot and record the dynamics on a training data-set."""
function take_snapshot(net::Net, log::Log, inputs::Array{Float64,2}, egomotion::Array{Float64,2}, s::Dict{String,Any},
    nTestElements::Int=64)

    l = s["presentationLength"]::Int
    fadeLength = s["fadeLength"]::Float64
    interval = s["snapshotLogInterval"]::Int

    nSteps = size(inputs, 1) * l
    nFirstSteps = min(nSteps, nTestElements*l)
    snapshot = create_snapshot(nFirstSteps, interval, log.t, inputs, net, s)

    v1_spikes = falses(net.n_v1)
    m2_spikes = falses(net.n_m2)
    rec_counter = 0
    for i in 1:nFirstSteps
        x = fade_images(inputs,i,s)
        y = fade_images(egomotion,i,s)
        step_net(net, x, y, s, update=false)

        rec = net.reconstruction
        if ((i - 1) % l > l * fadeLength)
            snapshot.reconstruction_means[div(i - 1, l) + 1, :] += rec / (l * (1.0 - fadeLength))
        end
        snapshot.reconstruction_vars[div(i - 1, l) + 1, :] +=
            (net.x_outputs - rec).^2 / (l - 1)

        v1_spikes .|= net.v1_spikes
        m2_spikes .|= net.m2_spikes

        if i % interval == 0
            ind = div(i, interval)
            snapshot.v1_spikes[ind,:] = v1_spikes
            snapshot.m2_spikes[ind,:] = m2_spikes
            v1_spikes = falses(net.n_v1)
            m2_spikes = falses(net.n_m2)
            snapshot.x_outputs[ind,:] = net.x_outputs
            snapshot.y_outputs[ind,:] = net.y_outputs
            snapshot.v1_inputs[ind,:] = net.v1_input
            snapshot.m2_inputs[ind,:] = net.m2_input
            snapshot.v1_outputs[ind,:] = net.v1_outputs
            snapshot.m2_outputs[ind,:] = net.m2_outputs
            snapshot.reconstructions[ind, :] = rec
            snapshot.reconstructions_y[ind, :] = net.ym2_weights' * net.m2_outputs
        end
    end

    push!(log.snapshots, snapshot)
end


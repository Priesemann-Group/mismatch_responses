using Statistics: mean, var
using Dates
""" The temp log can be used to save additional quantities.
They will be plotted automatically in the end. The 'runningLog'
is used to average over time until the next logging point is reached.

To log something:
1. create desired variable in runningLog & temp log in *setup_temp_log*
2. update runningLog every timestep in *update_running_log*
3. save results from averaging to the temp log in *log_temp_log*"""

function setup_temp_log(net::Net, nSteps::Int, interval::Int)::Dict{String, Any}
    runningLog = Dict{String, Any}()

    # time of log
    net.log.temp["t"] = zeros(div(nSteps, interval))

    # variance learned by the net
    net.log.temp["var"] = zeros(div(nSteps, interval))
    runningLog["var"] = 0.0

    # mean firing rate
    net.log.temp["firing_rates_v1"] = zeros(div(nSteps, interval), net.n_v1)
    runningLog["firing_rates_v1"] = zeros(net.n_v1)
    net.log.temp["firing_rates_m2"] = zeros(div(nSteps, interval), net.n_m2)
    runningLog["firing_rates_m2"] = zeros(net.n_m2)

    # mean z bias
    net.log.temp["v1_biases"] = zeros(div(nSteps, interval), net.n_v1)
    net.log.temp["m2_biases"] = zeros(div(nSteps, interval), net.n_m2)

    # mean xz weights
    net.log.temp["mean_xv1_weights"] = zeros(div(nSteps, interval))
    runningLog["mean_xv1_weights"] = 0.0

    # var xz weights
    net.log.temp["var_xv1_weights"] = zeros(div(nSteps, interval))
    runningLog["var_xv1_weights"] = 0.0

    # membrane
    net.log.temp["v1_inputs"] = zeros(div(nSteps, interval), net.n_v1)
    runningLog["v1_inputs"] = zeros(net.n_v1)

    # test measures
    net.log.temp["test_decoder_loss"] = zeros(div(nSteps, interval))
    net.log.temp["test_decoder_likelihood"] = zeros(div(nSteps, interval), 2)

    return runningLog
end

function update_running_log(net::Net, runningLog::Dict{String,Any},
    x_input::Array{Float64,1}, interval::Int, numSamples::Int, s::Dict{String,Any})

    t = net.log.t
    k = div(t - 1, interval) + 1

    runningLog["var"] += net.sigma_V1 ^ 2 / numSamples
    runningLog["mean_xv1_weights"] += mean(net.xv1_weights) / numSamples
    runningLog["var_xv1_weights"] += var(net.xv1_weights) / numSamples
    runningLog["v1_inputs"] += net.v1_input ./ numSamples
end

function log_temp_log(net::Net, runningLog::Dict{String,Any},
    inputs::Array{Float64,2}, egomotion::Array{Float64,2}, interval::Int, s::Dict{String,Any})

    t = net.log.t
    k = div(t - 1, interval) + 1
    dt = net.log.settings["dt"]
    temp = net.log.temp

    test_performance(net, inputs, egomotion, interval, s)

    temp["t"][k] = t

    temp["var"][k] = runningLog["var"]
    runningLog["var"] = 0.0

    temp["firing_rates_v1"][k, :] = runningLog["firing_rates_v1"] / (0.001 * dt)
    runningLog["firing_rates_v1"] = zeros(net.n_v1)
    temp["firing_rates_m2"][k, :] = runningLog["firing_rates_m2"] / (0.001 * dt)
    runningLog["firing_rates_m2"] = zeros(net.n_m2)

    temp["v1_biases"][k, :] = net.v1_biases
    temp["m2_biases"][k, :] = net.m2_biases

    temp["mean_xv1_weights"][k] = runningLog["mean_xv1_weights"]
    runningLog["mean_xv1_weights"] = 0.0

    temp["var_xv1_weights"][k] = runningLog["var_xv1_weights"]
    runningLog["var_xv1_weights"] = 0.0

    temp["v1_inputs"][k, :] = runningLog["v1_inputs"]
    runningLog["v1_inputs"] = zeros(net.n_v1)
end

""" Tests the performance on a training data-set."""
function test_performance(net::Net, inputs::Array{Float64,2}, egomotion::Array{Float64,2}, interval::Int, s::Dict{String,Any})
    l = s["presentationLength"]::Int

    nSteps = size(inputs, 1) * l
    k = div(net.log.t - 1, interval) + 1
    temp = net.log.temp

    for i in 1:nSteps
        x = fade_images(inputs,i,s)
        y = fade_images(egomotion,i,s)
        step_net(net, x, y, s, update=false)

        temp["test_decoder_loss"][k] += calc_decoder_loss(net, x) / nSteps
        temp["test_decoder_likelihood"][k,1] += log_decoder_likelihood(net) / nSteps
    end
end

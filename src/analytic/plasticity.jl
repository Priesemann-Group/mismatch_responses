
function update_net(net::Net, s::Dict{String, Any})
    update_xv1_weights(net, s)
    update_xm2_weights(net, s)
    update_ym2_weights(net, s)
    update_v1_biases(net, s)
    update_m2_biases(net, s)
end

function update_xv1_weights(net::Net, s::Dict{String,Any})
    eta = s["learningRateFeedForwardV1"]::Float64 * s["dt"]::Float64
    rec = net.reconstruction
    zv1 = net.v1_outputs
    x = net.x_outputs

    e = x - rec
    dF = e * zv1'
    net.xv1_weights += eta * dF'
end

function update_xm2_weights(net::Net, s::Dict{String,Any})
    eta = s["learningRateFeedForwardM2"]::Float64 * s["dt"]::Float64
    rec = net.reconstruction
    zm2 = net.m2_outputs
    x = net.x_outputs

    e = x - rec
    dF = e * zm2'
    net.xm2_weights += eta * dF'
end

function update_ym2_weights(net::Net, s::Dict{String,Any})
    eta = s["learningRateFeedForwardM2"]::Float64 * s["dt"]::Float64
    F = net.ym2_weights
    zm2 = net.m2_outputs
    y = net.y_outputs

    e = y - F' * zm2
    dF = e * zm2'
    net.ym2_weights += eta * dF'
end

function update_v1_biases(net::Net, s::Dict{String,Any})
    r = net.v1_rates
    b = net.v1_biases
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateHomeostaticBiasV1"]::Float64 * s["dt"]::Float64
    rho = s["rhov1"]::Float64
    dt = s["dt"]::Float64

    goalrate = dt * rho
    @inbounds for j in 1:net.n_v1
        db = goalrate - r[j]
        b[j] += eta * db
    end
    net.v1_rates .= 0.0
end

function update_m2_biases(net::Net, s::Dict{String,Any})
    r = net.m2_rates
    b = net.m2_biases
    batchmult = s["updateInterval"]::Int
    eta = batchmult * s["learningRateHomeostaticBiasM2"]::Float64 * s["dt"]::Float64
    rho = s["rhom2"]::Float64
    dt = s["dt"]::Float64

    goalrate = dt * rho
    @inbounds for j in 1:net.n_m2
        db = goalrate - r[j]
        b[j] += eta * db
    end
    net.m2_rates .= 0.0
end


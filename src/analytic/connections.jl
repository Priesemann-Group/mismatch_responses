using SparseArrays

function calc_v1_input(net::Net, s::Dict{String,Any})::Array{Float64,1}
    o2 = net.sigma_V1_2
    Fv1 = net.xv1_weights
    Fm2 = net.xm2_weights
    
    x = net.x_outputs
    zv1 = net.v1_outputs
    zm2 = net.m2_outputs
    
    inputs = net.v1_biases - o2 * 0.25 * diag(Fv1 * Fv1')
    inputs += o2 .* Fv1 * (x - Fv1' * zv1 - Fm2' * zm2)

    return inputs
end

function calc_m2_input(net::Net, s::Dict{String,Any})::Array{Float64,1}
    o2 = net.sigma_M2_2
    F = net.ym2_weights
    
    y = net.y_outputs
    zm2 = net.m2_outputs
    
    inputs = net.m2_biases - o2 * 0.25 * diag(F * F')
    inputs += o2 .* F * (y - F' * zm2)

    return inputs
end



function create_xv1_connections(s::Dict{String,Any})::Array{Float64,2}
    n_x = s["n_x"]::Int
    n_v1 = s["n_v1"]::Int
    wVar = s["weightVariance"]::Float64
    wMean = s["weightMean"]::Float64

    xv1_weights = randn(n_v1, n_x) * sqrt(wVar) .+ wMean
    xv1_weights = map(x -> max(0, x), xv1_weights)

    return xv1_weights
end

function create_xm2_connections(s::Dict{String,Any})::Array{Float64,2}
    n_x = s["n_x"]::Int
    n_m2 = s["n_m2"]::Int
    wVar = s["weightVariance"]::Float64
    wMean = s["weightMean"]::Float64

    xm2_weights = randn(n_m2, n_x) * sqrt(wVar) .+ wMean
    xm2_weights = map(x -> max(0, x), xm2_weights)

    return xm2_weights
end

function create_ym2_connections(s::Dict{String,Any})::Array{Float64,2}
    n_y = s["n_y"]::Int
    n_m2 = s["n_m2"]::Int
    wVar = s["weightVariance"]::Float64
    wMean = s["weightMean"]::Float64

    ym2_weights = randn(n_m2, n_y) * sqrt(wVar) .+ wMean
    ym2_weights = map(x -> max(0, x), ym2_weights)
    
    mid = Int(div(n_m2,2))
    ym2_weights[1:mid,1] .= 0.7 / n_m2
    ym2_weights[mid+1:end,1] .= -0.7 / n_m2

    return ym2_weights
end

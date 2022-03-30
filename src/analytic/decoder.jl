
function reconstruct_input(net::Net, s::Dict{String,Any})::Array{Float64,1}
    zv1 = net.v1_outputs
    Fv1 = net.xv1_weights
    zm2 = net.m2_outputs
    Fm2 = net.xm2_weights

    return Fv1' * zv1 + Fm2' * zm2
end

""" Calculates decoder loss in respect to the current network input"""
function calc_decoder_loss(net::Net)::Float64
    rec = net.reconstruction
    x = net.x_outputs

    err = x - rec
    return 0.5 / net.n_x * err' * err
end

""" Calculates decoder loss in respect to input x"""
function calc_decoder_loss(net::Net, x::Array{Float64,1})::Float64
    rec = net.reconstruction

    err = x - rec
    return 0.5 / net.n_x * err' * err
end

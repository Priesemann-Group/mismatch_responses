

function calc_var(net::Net)::Float64
    x = net.x_outputs
    rec = net.reconstruction

    M = x - rec
    return M.^2
end

function calc_var(net::Net, x_input::Array{Float64,1})::Float64
    x = x_input
    rec = net.reconstruction

    M = x - rec
    return M.^2
end

function log_decoder_likelihood(net::Net)::Float64
    x = net.x_outputs
    var = net.sigma_V1
    rec = net.reconstruction

    post = log_multivariate_gaussian(x, rec, var)
    return post 
end


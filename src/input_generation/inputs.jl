
function create_visual_flow_inputs(nPatterns::Int, s::Dict{String,Any}; test::Bool=false)::Tuple{Array{Float64,2},Array{Float64,2}}
    max0(x) = max(0, x)
    
    n = s["n_x"]
    p = s["inputMotionProbability"]::Float64
    
    @assert (n % 2 == 0) "Number of inputs has to be divisible by 2."

    dims = div(n,2)

    input = zeros(nPatterns, dims*2)
    egomotion = rand([-1,1],nPatterns,1) # left/steady/right
    randos_motion = rand(nPatterns, n) 
    
    
    for i in 1:nPatterns
        temp = ones(dims) .* egomotion[i]
        for j in 1:dims
            if (randos_motion[i,j] <= p/2) # left motion
                temp[j] -= 1
            elseif (randos_motion[i,j] >= 1.0 - p/2) # right motion
                temp[j] += 1
            end
        end
        
        input[i,1:dims] .= map(max0, temp)
        input[i,dims+1:end] .= map(max0, -temp)
    end
    
    if test
        egomotion[1:10,:] .= 1
        
        for i in 1:10
            temp = ones(dims) .* egomotion[i]
            if i == 3 || i == 5
                # left motion
                temp[2] -= 1
            end
            if i == 8
                # right motion
                temp[2] += 1
            end
            
            input[i,1:dims] .= map(max0, temp)
            input[i,dims+1:end] .= map(max0, -temp)
        end
    end
    
    egomotion = hcat(map(max0, egomotion), map(max0, -egomotion))
    
    return input, egomotion
end

function fade(inputs::SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},
 old_inputs::SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}, 
 t::Float64, fadeTime::Float64)::Array{Float64,1}
    f(t) = min(1.0,max(0.0,t / fadeTime))
    return inputs .* f(t) + old_inputs .* (1.0 - f(t))
end

function fade_images(x_inputs::Array{Float64,2},t::Int,s::Dict{String,Any})::Array{Float64,1}
    l = s["presentationLength"]::Int
    fadeLength = s["fadeLength"]::Float64
    
    indImg = div(t - 1,l) + 1

    inp = view(x_inputs, indImg, :)
    old_inp = view(x_inputs, max(1, indImg-1), :)
    x = fade(inp, old_inp, (((t-1) % l) + 1) / l, fadeLength)

    return x
end


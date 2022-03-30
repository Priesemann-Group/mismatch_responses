using MultivariateStats
using DelimitedFiles: readdlm, writedlm

""" Separates subimages in image with border for plotting"""
function separate_subimages(X::Array{Float64, 2}, sub_length_x::Int, sub_length_y::Int, border_width::Int; inverted=false)
    bw = border_width
    splits = split_images(X, sub_length_x, sub_length_y)
    s = size(splits)
    X_new = zeros(s[1], s[2]+2*bw, s[3]+2*bw)
    if !inverted
        X_new .= minimum(X) - 0.1
    else
        X_new .= maximum(X)
    end
    X_new[:,bw+1:end-bw,bw+1:end-bw] = splits
    return stitch_images(X_new)
end

""" Splits image into square tiles of a given side-length for single image"""
function split_images(X::Array{Float64, 2}, split_length_x::Int, split_length_y::Int)
    image_slX = size(X,1)
    image_slY = size(X,2)

    slx = split_length_x
    sly = split_length_y
    splitsX = div(image_slX, slx)
    splitsY = div(image_slY, sly)
    temp = zeros(splitsX * splitsY, slx, sly)

    i = 0
    for sx in 1:splitsX
        for sy in 1:splitsY
            i += 1
            cut = X[slx*(sx-1)+1:slx*sx, sly*(sy-1)+1:sly*sy]
            temp[i,:,:] = cut
        end
    end
    return temp
end

function stitch_images(X::Array{Float64,3})
    m = Int(sqrt(size(X,1)))
    slx = size(X,2)
    sly = size(X,3)

    temp = zeros(m * slx, m * sly)

    i = 0
    for sx in 1:m
        for sy in 1:m
            i += 1
            cut = X[i, :, :]
            temp[slx*(sx-1)+1:slx*sx, sly*(sy-1)+1:sly*sy] = cut
        end
    end
    return temp
end

""" Whiten data using a whitening matrix computed with the Stats package.
 - W: Whitening matrix, if already calculated"""
function whiten_images_cholesky(X::Array{Float64,2}, W=nothing)
    X = permutedims(X, [2,1])
    if isnothing(W)
        f = fit(Whitening, X, mean=0)
        W = f.W
    end
    X = W' * X
    return W, permutedims(X, [2,1])
end



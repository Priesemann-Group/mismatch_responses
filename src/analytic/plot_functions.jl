using Statistics: mean
import JSON
import PyPlot
using Plots
pyplot()

""" Plots net into 'name'. Folder 'name' has to exist! """
function plot_net(net::Net, name::String)
    folder = name * "/"

    open(folder * "settings.json","w") do f
        json_string = JSON.json(net.log.settings, 4)
        write(f, json_string)
    end

    make_plots(net, folder)
end

""" Folder 'name' has to exist! """
function plot_patterns(patterns::Array{Float64, 2}, name::String, s::Dict{String,Any})
    folder = name * "/patterns/"

    mkdir(folder)
    string = folder * "patterns.svg"
    
    xlength = Int(div(size(patterns)[2],2))
    ylength = 2
    
    plot_images(patterns, xlength, ylength, string)
end

function make_plots(net::Net, folder::String)
    log = net.log
    dt = net.log.settings["dt"]

    # plot temporally saved variables
    for key in keys(log.temp)
        if (key != "t")
            plot_temp(log.temp[key], dt .* log.temp["t"], folder * "temp_" * key)
        end
    end

    # plot snapshots
    for i in 1:length(log.snapshots)
        print("Plotting snapshot $i\n")
        plot_snapshot(net, log.snapshots[i], folder, log.settings)
    end
end

function plot_snapshot(net::Net, snapshot::Snapshot, folder::String, s::Dict{String,Any})
    t = snapshot.t
    ap = ""
    try
        mkdir(folder * "snapshot-$t")
    catch e
        try 
            mkdir(folder * "snapshot2-$t")
            ap = "2"
        catch e
        end
    end
    plot_output(snapshot.x_outputs[:,1:min(4,s["n_x"])], folder * "snapshot$ap-$t/x_output", s)
    plot_output(snapshot.y_outputs[:,1:min(4,s["n_y"])], folder * "snapshot$ap-$t/y_output", s)
    plot_spikes(snapshot.v1_spikes, folder * "snapshot$ap-$t/v1_spikes", s)
    plot_spikes(snapshot.m2_spikes, folder * "snapshot$ap-$t/m2_spikes", s)
    plot_output(snapshot.v1_inputs[:,1:min(4,s["n_v1"])], folder * "snapshot$ap-$t/v1_input", s)
    plot_output(snapshot.m2_inputs[:,1:min(4,s["n_m2"])], folder * "snapshot$ap-$t/m2_input", s)
    plot_output(snapshot.v1_outputs[:,1:min(4,s["n_v1"])], folder * "snapshot$ap-$t/v1_output", s)
    plot_output(snapshot.m2_outputs[:,1:min(4,s["n_m2"])], folder * "snapshot$ap-$t/m2_output", s)
    plot_weights(snapshot.xv1_weights, folder * "snapshot$ap-$t/xv1_weights", s)
    plot_weights(snapshot.xm2_weights, folder * "snapshot$ap-$t/xm2_weights", s)
    plot_1dweights(snapshot.ym2_weights, folder * "snapshot$ap-$t/ym2_weights", s)
    plot_biases(snapshot.v1_biases, folder * "snapshot$ap-$t/v1_biases")
    plot_biases(snapshot.m2_biases, folder * "snapshot$ap-$t/m2_biases")
    plot_reconstructions(snapshot.reconstruction_means, folder * "snapshot$ap-$t/", s)
    plot_reconstructions(snapshot.reconstruction_vars, folder * "snapshot$ap-$t/", s, "_var")
    plot_reconstruction_comparison(snapshot.reconstructions, snapshot.x_outputs, folder * "snapshot$ap-$t/", "_x", s)
    plot_reconstruction_comparison(snapshot.reconstructions_y, snapshot.y_outputs, folder * "snapshot$ap-$t/", "_y", s)
end

function plot_temp(temp, ts, name::String)
    plot(ts, temp, linewidth=1.5, grid=false, legend=false)
    savefig(name * ".svg")
end

function plot_reconstructions(reconstructions::Array{Float64}, name::String,
    s::Dict{String,Any}, app::String="")
    
    folder = name * "reconstructions/"
    mkpath(folder)
    string = folder * "reconstruction" * app * ".svg"
    
    xlength = Int(div(size(reconstructions)[2],2))
    ylength = 2
    
    plot_images(reconstructions, xlength, ylength, string)
end

function plot_reconstruction_comparison(recs::Array{Float64}, xs::Array{Float64},
    name::String, appendix::String, s::Dict{String,Any})
    
    if appendix=="_x"
        n = s["n_x"]
    elseif appendix=="_y"
        n = s["n_y"]
    end

    dt = s["dt"]
    folder = name * "reconstructions/"
    mkpath(folder)
    tmax = min(Int(floor(10*s["presentationLength"]/s["snapshotLogInterval"])),size(recs,1))
    vars = mean((recs - xs).^2, dims=1)
    inds1 = sortperm(reshape(vars,:),lt=(>))
    inds2 = sortperm(reshape(mean(xs, dims=1),:),lt=(>))
    if n > 4
        inds = cat(inds1[1:2], inds1[end-1:end], inds2[1:2], dims=1)
    else
        inds = 1:n
    end
    for i in 1:length(inds)
        plot(dt .* collect(1:tmax) .* s["snapshotLogInterval"],
             cat(recs[1:tmax,inds[i]], xs[1:tmax,inds[i]], dims=2),
             linewidth=1.5, grid=false, legend=false, size=(1200,400));
        savefig(folder * "comparison$(appendix)$i.svg")
    end
end

function plot_spikes(spikes::Array{Bool}, name::String, s::Dict{String,Any})
    dt = s["dt"]
    data = []
    isis = []
    for n in 1:size(spikes)[2]
        sub = []
        last_t = 1
        for t in 1:min(Int(floor(10*s["presentationLength"]/s["snapshotLogInterval"])),size(spikes)[1])
            if spikes[t,n]
                push!(sub, dt*t*0.001*s["snapshotLogInterval"])
                push!(isis, dt*(t - last_t)*0.001*s["snapshotLogInterval"])
                last_t = t
            end
        end
        push!(data, sub)
    end
    PyPlot.clf()
    PyPlot.figure(figsize=(12.0,4.0))
    PyPlot.eventplot(data);
    PyPlot.savefig(name * ".svg")
    PyPlot.close_figs()

    #PyPlot.clf()
    #PyPlot.hist(isis, bins=30, range=(0.0,0.05));
    #PyPlot.savefig(name * "_ISI.svg")
    #PyPlot.close_figs()
end

function plot_output(output::Array{Float64,2}, name::String, s::Dict{String,Any})
    dt = s["dt"]
    tmax = min(Int(floor(10*s["presentationLength"]/s["snapshotLogInterval"])),size(output,1))
    ts = collect(1:tmax)*dt*0.001
    output = output[1:tmax,:]
    plot(ts,output,title="Outputs",xlabel="\$t\$",ylabel="\$z(t)\$");
    savefig(name * ".svg")
end

function plot_weights(weights::Array{Float64}, name::String, s::Dict{String,Any})
    

    xlength = Int(div(size(weights)[2],2))
    ylength = 2
    string = name * ".svg"
    
    plot_images(weights, xlength, ylength, string)
end

function plot_1dweights(weights::Array{Float64}, name::String, s::Dict{String,Any})
    xlength = size(weights)[2]
    ylength = 1
    string = name * ".svg"
    
    plot_images(weights, xlength, ylength, string)
end


function plot_biases(biases, name::String)
    heatmap(reshape(biases,1,:));
    savefig(name * ".svg")
end

function plot_images(images::Array{Float64, 2}, xlength, ylength, name::String)
    nImages = min(64,size(images)[1])
    nImg = size(images)[2]
    nCols = Int(ceil(sqrt(nImages)))

    img = zeros(nCols*xlength,nCols*ylength)
    for i in 1:nImages
        pattern = images[i,:]
        col = ((i-1)%nCols)
        row = div(i-1,nCols)
        img[col*xlength+1:(col+1)*xlength, row*ylength+1:(row+1)*ylength] =
            reshape(pattern,xlength,ylength)
    end
    img = separate_subimages(img, xlength, ylength, 2)
    maxval = max(maximum(img),-minimum(img))
    PyPlot.clf()
    PyPlot.imshow(img',cmap="RdBu", vmin=-maxval, vmax=maxval)
    ax = PyPlot.gca()
    ax.axis("off")
    PyPlot.colorbar()
    PyPlot.savefig(name)
    PyPlot.close_figs()
end


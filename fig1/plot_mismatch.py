import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil
sys.path.insert(0, '../src/')


def main():
    ARGS = sys.argv
    if (len(ARGS) <= 1):
        print("Please provide the file to plot as argument.")
        return
    
    file = ARGS[1]

    print("Plotting...")
    
    dic = h5py.File(file, 'r')
    temp = dic[u'temp']
    meta = temp.attrs
    
    dt = temp.attrs.get("dt") 
    interval = temp.attrs.get("snapshotLogInterval")
    pres_len = int(temp.attrs.get("presentationLength") / temp.attrs.get("snapshotLogInterval"))

    snapshots = [dic[k] for k in sorted(dic.keys()) if 'snapshot' in k]

    snap = snapshots[-1]
    
        
    def put_spikes_in_traces(spikes, traces, spike_height):
        for n in range(spikes.shape[0]):
            s = spike_height + np.max(traces[:,n])
            for t in range(spikes.shape[1]):
                ind_t = int(spikes[n,t]/dt)
                if spikes[n,t]>=0 and ind_t < traces.shape[0]:
                    traces[ind_t,n] = s 
        
    
    
    tmin = 0
    tmax = int(pres_len * 10)
    spike_height = 50
    
    
        
    spikes = snap.get('v1_spikes')[()]
    inputs = snap.get('v1_inputs')[()].T
    put_spikes_in_traces(spikes, inputs, spike_height)
    
    
    m2_spikes = snap.get('m2_spikes')[()]
    m2_inputs = snap.get('m2_inputs')[()].T
    put_spikes_in_traces(m2_spikes, m2_inputs, spike_height)
    
        
    flow = snap.get('x_outputs')[()].T
    n = flow.shape[1] // 2
    flow = flow[:int(tmax*interval),:n] - flow[:int(tmax*interval),n:]
    
    
    scale = 0.8
    fig = plt.figure(figsize=(12*scale, 11*scale))
    plt.axis('off')
    
    N = spikes.shape[0] + m2_spikes.shape[0] + 1
    
    axs = []
    ax = plt.subplot2grid((N, 1), (0, 0), colspan=1, rowspan=1)
    axs.append(ax)
    for ind in range(N-1):
        ax = plt.subplot2grid((N, 1), (ind, 0), colspan=1, rowspan=1, sharex=axs[0])
        axs.append(ax)

    for ind in range(spikes.shape[0]):
        ind2 = ind
        axs[ind2].plot(dt * interval * np.arange(tmax-tmin) / 1000, inputs[tmin:tmax,ind], 
                 color='cornflowerblue', label=r"$\hat x$")
        handles, labels = axs[ind2].get_legend_handles_labels()
        #axs[ind2].set_ylabel('membrane voltage')
        axs[ind2].set_ylim([-45,55])
        axs[ind2].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        for side in axs[ind2].spines:
            if side != "left":
                axs[ind2].spines[side].set_visible(False)
        fig.add_subplot(axs[ind2])
        
        
    for ind in range(m2_spikes.shape[0]):
        ind2 = spikes.shape[0] + ind
        axs[ind2].plot(dt * interval * np.arange(tmax-tmin) / 1000, m2_inputs[tmin:tmax,ind], 
                 color='cornflowerblue', label=r"$\hat x$")
        handles, labels = axs[ind2].get_legend_handles_labels()
        #axs[ind2].set_ylabel('membrane voltage')
        axs[ind2].set_ylim([-45,55])
        axs[ind2].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        for side in axs[ind2].spines:
            if side != "left":
                axs[ind2].spines[side].set_visible(False)
        fig.add_subplot(axs[ind2])
        
    
    #### flow
    
    ind_flow = 1
    
    axs[-1].plot(dt * interval * np.arange(tmax-tmin) / 1000, flow[tmin:tmax,ind_flow], 
             color='cornflowerblue', label=r"$\hat x$")
    handles, labels = axs[ind2].get_legend_handles_labels()
    #axs[-1].set_ylabel('flow')
    axs[-1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    for side in axs[-1].spines:
        if side != "left":
            axs[-1].spines[side].set_visible(False)
    fig.add_subplot(axs[-1])
    
    plt.tight_layout()
    plt.savefig("mismatch.svg".format(ind), dpi=500)


main()


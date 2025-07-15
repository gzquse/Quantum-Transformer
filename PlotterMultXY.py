__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys
sys.path.append(os.path.abspath("../cloud_job/"))
from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap


#...!...!....................
def compute_correlation_and_draw_line(ax, x_data, y_data, xLR=[], showSummary=True):
    """Compute correlation and draw a line at the angle of correlation."""
    correlation = np.corrcoef(x_data, y_data)[0, 1]

    # Line representing correlation - slope based on correlation
    # y = mx + c, where m is the correlation coefficient
    # We pass through the mean of the points for the line of best fit
    mean_x, mean_y = np.mean(x_data), np.mean(y_data)
    ax.plot(mean_x,mean_y,'o',ms=6, color='#4A90A4', markeredgecolor='white', markeredgewidth=1)
    m = correlation * np.std(y_data) / np.std(x_data)
    c = mean_y - m * mean_x
    
    # Points for the line
    x12 = np.array([min(x_data), max(x_data)])
    y12 = m * x12 + c
    ax.plot(x12, y12, '--', lw=2.0, color='#5D737E', alpha=0.8)

    if showSummary:
        th=np.arctan(m)
        txt='correl: %.2f,   theta %.0f deg'%(correlation,th/np.pi*180)
        ax.text(0.05,0.92,txt,transform=ax.transAxes,color='black')    
    
    ax.grid(True, alpha=0.3, color='#B0C4DE')
    return 

        
#...!...!....................
def plot_histogram(ax, res_data, showSummary=True):
    """Plot histogram of the difference and annotate mean and std."""
    
    ax.hist(res_data, bins=25, color='#87CEEB', alpha=0.7, edgecolor='white', linewidth=0.5)
    mean = np.mean(res_data)
    std = np.std(res_data)
    # assuming normal distribution, compute std error of std estimator
    # SE_s=std/sqrt(2(n-1)), where n is number of samples
    N=res_data.shape[0]
    se_s=std/np.sqrt(2*(N-1))
    ax.axvline(mean, color='#4A90A4', linestyle='dashed', linewidth=2)
    
    if showSummary:
        txt='Mean: %.3f\nRMSE: %0.3f +/- %0.3f'%(mean,std,se_s)
        ax.annotate(txt, xy=(0.05, 0.85), c='black', xycoords='axes fraction')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))


#...!...!....................
def summary_column(md):
    #pprint(md)
    pmd=md['payload']
    smd=md['submit']
    tmd=md['transpile']
    pom=md['postproc']
    txt=md['short_name']
    txt+='\nback: %s'%smd['backend']
    txt+='\n'+md['job_qa']['timestamp_running']
    txt+='\nshots/addr : %d'%(smd['num_shots']/pmd['num_addr'])
    txt+='\nshots/img : %d k'%(smd['num_shots']/1000)
    txt+='\nnum sample %d'%(pmd['num_sample'])
    txt+='\nsample size: %d'%(pmd['seq_len'])
    txt+='\nnum addr: %d'%pmd['num_addr']
    txt+='\nqubits: %d'%len(tmd['phys_qubits'])
    if 'ibm' in smd['backend']:
        txt+='  RC: '+ ('T ' if smd['random_compilation'] else 'F ')
        useDD=False
        if 'dynamical_decoupling' in smd: useDD=smd['dynamical_decoupling'] # patch for the past
        txt+='  DD: '+ ('T' if useDD else 'F')
    txt+='\nnum 2q gates: %d'%tmd['2q_gate_count']
    txt+='\n2q gates depth: %d'%tmd['2q_gate_depth']

    txt+='\nhwCal: %s'%pom['hw_calib']
    if pom['hw_calib']: txt+=': %.2f'%pom['ampl_fact']
    
    return txt
    if 'noise_model' in smd:
        txt+='\nfake : %s'%(smd['noise_model'])       
 
   
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def ehands_accuracy(self,bigD,md,figId=1, asCol=False, showSummary=True):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']
        tmd=md['transpile']
        pom=md['postproc']
        if 'truth_rangeLR' in md:
            xrL,xrR=md['truth_rangeLR']
        else:
            xrL,xrR=-1.15, 1.15

        resMX=md['plot']['resid_max_range']
        
        figId=self.smart_append(figId)        
        nrow,ncol=1,3 ; xyIn=(10,3)
        if asCol:
            nrow,ncol=3,1 ; xyIn=(5,10)
        if showSummary:
            xyIn=(12,3)
        fig=self.plt.figure(figId,facecolor='white', figsize=xyIn)
        
        topTit=[ 'job: '+md['short_name'], 'Residual ',smd['backend']]

        
        #....... plot data .....
        rdata=bigD['rec_udata'].flatten()
        tdata=bigD['true_output'].flatten()
        #....  left column ....
        ax = self.plt.subplot(nrow,ncol,1)
           
        ax.scatter(tdata,rdata,alpha=0.6,s=8, c='#7FB3D3', edgecolors='white', linewidth=0.2)
        ax.set(xlabel='true value',ylabel='reco')
        compute_correlation_and_draw_line(ax, tdata, rdata, showSummary=showSummary)
        ax.set_aspect(1.)
        ax.set_xlim(xrL,xrR);ax.set_ylim(xrL,xrR)
        x12 = np.array([min(tdata), max(tdata)])
        ax.plot(x12,x12,ls='--',c='#708090',lw=1.5, alpha=0.7)           

        if showSummary:
            ax.set_title(topTit[0], color='#2E5266') 
            txt='\nhwCal: %s:  %.2f'%(pom['hw_calib'],pom['ampl_fact'])
            ax.text(0.32, 0.21, txt, fontsize=10, color='black', ha='left', va='top',transform=ax.transAxes)
        
        #..... right column ....
        ax = self.plt.subplot(nrow,ncol,3)
        res_data = rdata - tdata
        h = ax.hist2d(rdata, res_data, bins=20, cmap='Blues',cmin=0.1)
        self.plt.colorbar(h[3], ax=ax)

        compute_correlation_and_draw_line(ax, rdata , res_data, showSummary=showSummary) 
        ax.axhline(0.,ls='--',c='#708090',lw=1.5, alpha=0.7)

        ax.set_ylabel('reco-true')
        ax.set(xlabel='reco value',ylabel='reco-true')
        
        ax.set_xlim(xrL,xrR); ax.set_ylim(-resMX,resMX)
        ax.grid(True, alpha=0.3, color='#B0C4DE')
        if showSummary or 'ibm' in smd['backend']: 
            ax.set_title(topTit[1])
            txt='phys:%s'%(tmd['phys_qubits'])
            ax.text(0.01, 0.1, txt, fontsize=10, color='black', ha='left', va='top',transform=ax.transAxes)
 
        #..... middle column ....
        ax = self.plt.subplot(nrow,ncol,2) 
        plot_histogram(ax, res_data, showSummary=showSummary)
        xLab= 'reco-true'
        ax.set(xlabel=xLab,ylabel='num pixels')
        ax.axvline(0.,ls='--',c='#708090',lw=1.5, alpha=0.7)
        ax.set_xlim(-resMX,resMX)
        ax.grid(True, alpha=0.3, color='#B0C4DE')
        
        # .... decorations ....
        # Overlay the text on top of the plots
        if showSummary:
            ax.set_title(topTit[2], color='#2E5266')
            txt=summary_column(md)
            ax.text(0.80, 0.95, txt, fontsize=10, color='black', ha='left', va='top',transform=ax.transAxes)

#...!...!..................
    def truth_only(self,bigD,md,figId=3):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']
        tmd=md['transpile']

        figId=self.smart_append(figId)        
        nrow,ncol=1,3
        fig, axs = self.plt.subplots(nrow,ncol, figsize=(10, 3.5), num=figId)

        data=bigD['inp_udata']

        xV=data[:,0].flatten()
        yV=data[:,1].flatten()
        xyV=xV*yV
        
        axs[0].hist(xV, bins=50, range=(-1, 1), color='#A8D8B9', alpha=0.8, edgecolor='white', linewidth=0.5)
        axs[0].set_title('Histogram of input $x$', color='#2E5266')
        axs[0].set_xlabel('$x$')
        axs[0].set_ylabel('Count')
        axs[0].grid(True, alpha=0.3, color='#B0C4DE')

        axs[1].hist(yV, bins=50, range=(-1, 1), color='#D4A8A8', alpha=0.8, edgecolor='white', linewidth=0.5)
        axs[1].set_title('Histogram of input  $y$', color='#2E5266')
        axs[1].set_xlabel('$y$')
        axs[1].set_ylabel('Count')
        axs[1].grid(True, alpha=0.3, color='#B0C4DE')
        
        axs[2].hist(xyV, bins=50, range=(-1, 1), color='#B8A8D4', alpha=0.8, edgecolor='white', linewidth=0.5)
        axs[2].set_title('Histogram of  true $x \\times y$', color='#2E5266')
        axs[2].set_xlabel('$x \\times y$')
        axs[2].set_ylabel('Count')
        axs[2].grid(True, alpha=0.3, color='#B0C4DE')
# This scripts performs a Kmeans clustering on AML v2 TF values
# Data set in use is from AML immuno small molecule v2 cohort

"""Import packages"""
import pandas as pd
from os.path import join

"""Global variables"""
channelmarkerdic = {'C2':'cancer','C3':'proliferate','C4':'apoptosis','C5':'immune'}

"""Functions"""

def loadrunoutput():
    """Import packages"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    """Load data"""
    # Data path
    filedir = r'/home/cloud/users'
    filename = r'filename.csv'
    filepath = join(filedir,filename)
    # Load data
    # Load baseline corrected data for kmeans clustering
    df = pd.read_csv(filepath)
    # Drop the index column
    df = df.drop(['Unnamed: 0'],axis=1)

    """Select patients"""
    patients=['p8']
    df = df[df.patient.isin(patients)]

    """Standard scaling of data"""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[['TF_C2','TF_C3','TF_C4','TF_C5']].values),columns=['TF_C2','TF_C3','TF_C4','TF_C5'])

    """KMeans"""
    # Build kmeans class and train
    # Choose cluster number using elbow method
    # Keep random state 42
    km = KMeans(n_clusters=12, n_init=30, random_state=42)
    # Fit with all data
    kmfit = km.fit(df_scaled.values)
    # Save the cluster centers (12 clusters)
    centers = kmfit.cluster_centers_
    # Save the labels
    kmeansres = kmfit.labels_

    """Output result to csv"""
    # Append kmeans results to dfraw (normalized df before scaling)
    df.insert(len(df.columns),'km',kmeansres)
    # Output as csv file
    outdir = r'/home/cloud/users'
    outname = r'output.csv'
    outpath = join(outdir,outname)
    df.to_csv(outpath)

"""Function to calculate number of cells in each kmeans groups and in each drug well"""
def calculate_output_numofcell_in_kmcluster_in_well(fdir, fname):
    # This function calculate number of cells and output to csv
    # The output has drug wells as row and kmeans groups as column
    # the value is number of cells in that kmeans group and in that drug wells
    # Input - fdir: file directory
    #       - fname: file name (the csv file that contains kmeans output results)

    """Load data"""
    fpath = join(fdir,fname)
    df = pd.read_csv(fpath)
    # Drop indexing column
    df = df.drop(['Unnamed: 0'], axis=1)
    # Get drug types
    drugtypes = list(df['drugs'].unique())
    # Get kmeans cluster types and sort them
    # kmeans cluster usually is last column
    clustertypes = list(df[df.columns[-1]].unique())
    clustertypes.sort()
    # Create a df to save number of cells in one cluster group in one drug well
    df_numcells = pd.DataFrame(data=None,index=drugtypes,columns = ['G'+str(i) for i in clustertypes])
    df_percofcells = pd.DataFrame(data=None,index=drugtypes,columns = ['G'+str(i) for i in clustertypes])

    """Process find number of cells in each category (per drug and per cluster)"""
    # Loop one drug at a time
    for w in drugtypes:
        # Loop one cluster at a time
        for c in clustertypes:
            # Count number of cells that match the well and the cluster
            df_numcells.loc[w]['G'+str(c)]= df[(df['drugs'] == w)&(df[df.columns[-1]] == c)].shape[0]
    
    """Calculate total of one well and total of one kmeans group"""
    # Sum across row and add a new column
    df_numcells['ConditionTotal'] = df_numcells.sum(axis=1)
    # Sum acros column and add a rwo
    df_numcells.loc['GroupTotal'] = df_numcells.sum()
    # % of cells of each kmeans group
    df_numcells.loc['GroupPerc'] = (df_numcells.loc['GroupTotal'])/(df_numcells.loc['GroupTotal']['ConditionTotal'])

    """Calculate % of cells"""
    for w in drugtypes:
        df_percofcells.loc[w] = df_numcells.loc[w] / df_numcells.loc[w]['ConditionTotal']

    """Combine number of cells and % of cells"""
    dfout = pd.concat([df_numcells,df_percofcells],axis=0,sort=False)
    
    """Output to csv"""
    outname = 'kmsum_' + fname
    outpath = join(fdir,outname)
    dfout.to_csv(outpath)
"""end of function"""

"""Function to plot marker intensity across kmeans groups"""
def barplot_markerTF_kmgroup(fdir,fname,ch,title,yscale=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # This function creates a barplot of marker TF intensity on y and km groups on x
    # Input - fdir: file directory
    #       - fname: file name (the csv file that contains the kmeans results)
    #       - ch: marker channel, option ('C2','C3','C4','C5')
    #       - yscale: a list of y (TF) scale range [y1,y2]
    #
    """Load data"""
    fpath = join(fdir,fname)
    df = pd.read_csv(fpath)
    # Drop indexing column
    df = df.drop(['Unnamed: 0'], axis=1)
    # Get clusters and sort it in order
    # km cluster results is usually the last column
    clusterorders = list(df[df.columns[-1]].unique())
    clusterorders.sort()
    # Create the barplot
    marker = 'TF_'+ch
    ax = sns.boxplot(x=df.columns[-1], y=marker, data=df, order=clusterorders,color='white',showfliers=True,fliersize=2)
    ax.axhline(1, 0, 1)
    #sns.boxplot(x='km12init30run1', y='TF_C5', data=df, order=clusterorders, fliersize=2)
    plt.xlabel('km clusters')
    plt.ylabel('TF '+ ch + ' ' + channelmarkerdic[ch])
    plt.title(title)
    if yscale != None:
        plt.ylim(yscale)

"""Execution"""
def main():
    fdir = r'C:\Users\km'
    fname = r'km.csv'
    barplot_markerTF_kmgroup(fdir,fname,'C2','P1')

if __name__ == '__main__':
    main()
# This class "Combine" contains functions that combines dataframes for data curation purpose
# This class is built on top of "Combine_df.py"
# Author: Yunqi Yan

"""import packages"""
import os
import pandas as pd
import numpy as np
import re

class Combine:
    # Function: combine and output extracted wellexp pos/neg cells
    def combine_output_wellexp_positive_negative_cells(self, maindir):
        # This code combines well explorer labeld positive and negative cells of 4 markers into one csv
        # Initially, the data are like this:
        # One csv named e.g. "W1.csv" contains positive/negative cell labels of one condition of one patient
        # One patient folder contains 46 csv files coressponding to 46 conditions "W1.csv" to "W46.csv"
        # One main directory folder contains all patient folders e.g. 12 patient folders

        # This code combines them together into one csv, which contains:
        # Each row is a single cell, labeled by drug condition and patient
        # Colummn: immune_pos,cancer_pos,proliferate_pos,apoptosis_pos,
        #          immune (immnue marker raw TF),cancer (cancer raw TF),proliferate,apoptosis

        # Re-run this code may require modification of customized parameters defined in main function

        # Create an empty df to save all wells all patients data
        df_combined = pd.DataFrame(columns=['immune_pos','cancer_pos','proliferate_pos','apoptosis_pos','immune','cancer','proliferate','apoptosis','drugs','patient'])
        # Find patient folder
        patientfolder = [f for f in os.listdir(maindir) if '-' in f]
        # For loop one patient at a time
        for i in range(0,len(patientfolder)):
            patientdir = os.path.join(maindir, patientfolder[i])
            #patientdir = os.path.join(maindir,os.listdir(maindir)[i])
            # For loop one well at a time
            for w in range(0,len(os.listdir(patientdir))):
                filepath = os.path.join(patientdir,'W'+str(w+1).zfill(2)+'.csv')
                # Load data
                df = pd.read_csv(filepath)
                # Drop indexing column
                df = df.drop(['Unnamed: 0'], axis=1)
                # Add drug and patient info
                df['drugs']='W'+str(w+1)
                #df['patient'] = os.listdir(maindir)[i]
                df['patient'] = patientfolder[i]
                # Append df (this patient this drug) to df_combined
                df_combined = df_combined.append(df,sort=False)
            print('Complete '+patientfolder[i])
        df_combined.to_csv(os.path.join(maindir,'filename.csv'))
    # End of function

    # Function: combine multiple given csv into one csv
    def combine_ncsv_to_1csv_colstack(self,maindir,outdir):
        # This function combines multiple csv (ncsv) to one combine csv (1csv) along column stack
        # this function assumes that all given csv have same # of columns, maybe different # of rows
        # the function simply combines multiple csv along rows, maintaining same # of columns
        # Input - maindir: a list of multiple csv path
        #       - outdir: output directory that saves the output combined csv
        # Output - a combine csv
        # === Start of function =====
        # Create an empty combine df
        df_combined = pd.DataFrame()
        # Create an empty list to save repetaed csv file name
        csvnamelist = []
        # For loop one csv at a time
        for i in range(0,len(maindir)):
            path = maindir[i]
            csvname = os.path.basename(path)
            # Load one csv at a time
            df = pd.read_csv(path)
            df_combined = df_combined.append(df)
            csvnamelist = csvnamelist + [csvname]*(df.shape[0])
        # Insert csv file name
        df_combined.insert(0,'File',csvnamelist)
        # Output path
        outname = 'Filename.csv'
        outpath = os.path.join(outdir,outname)
        df_combined.to_csv(outpath)
    # End of function

    # Function: combine wellexp called pos/neg cells from selected patient and drug
    def combine_wellexp_posneg_selected_pt_dr(self):
        # This function select certain patients and drugs from a given file
        # The file locates is given
        # Certain patients and drugs are selected that match the annotated image
        # The output is wellexp called positive/negative cells
        # This is for Knight Pilot machine learning task to compare "major vote of human annotation" and "wellexp calls"
        # === Start of script =============
        # File path
        fdir = r'C:\Users\Input\input.csv'
        fname = 'filename.csv'
        fpath = os.path.join(fdir,fname)
        # Load data
        df = pd.read_csv(fpath)
        # Create an empty df_combined to save combined data
        df_combined = pd.DataFrame()
        # Select data
        # Create a dictionary of patient, well, starting cell id and ending cell id
        select_dic = {'m01904':['ID1','W1',5228,7717],
                    'm10196':['ID2','W10',4831,6071],
                    'm18730':['ID3','W18',6285,8910],
                    'm20002':['ID4','W20',1,1709],
                    'm22195':['ID5','W22',6953,7956]}
        # For loop one patient at a time
        for key in select_dic.keys():
            pt = select_dic[key][0]
            well = select_dic[key][1]
            start_id = select_dic[key][2]
            end_id = select_dic[key][3]
            df_selected = df[(df.patient==pt)&(df.drugs==well)&(df['Unnamed: 0'].isin(np.arange(start_id-1,end_id).tolist()))]
            df_combined = df_combined.append(df_selected)
        # Output df_cobined to csv
        outdir = r'C:\Users\Output'
        outname = 'output.csv'
        outpath = os.path.join(outdir,outname)
        df_combined.to_csv(outpath)
    # End of function

# Function: combine knight pilot DL prediction results multiple channel into on csv of one well
def combine_multichannel_1well_to_1well_post_DLwYH(self,inputdir,outdir):
    # This code specifically applies to csv that are direct output of knight pilot deep learning prediction code
    # Deep learning prediction code locates: C:\Users\...\prediction.py
    # Before combine, one csv file corresponds to one channel of one well;
    # After combine, one csv file corresponds to all channels of one well;
    # === Start of script ==============
    # channel marker dict
    ch_marker_dict = {'ch2':'cancer',
                    'ch3':'proliferate',
                    'ch4':'aptosis',
                    'ch5':'immune'}
    
    # Get all sub files
    filenames = os.listdir(inputdir)
    # Find unique conditions
    wells = [item[9:12] for item in filenames]
    uniquewells = list(set(wells))
    uniquewells.sort()

    # Combine all channels of one well into one csv
    for w in uniquewells:
        # Create an empty df to combine
        dfcombine = pd.DataFrame()
        matched_fname = [f for f in filenames if w in f]
        for fname in matched_fname:
            ch = fname[13:16]
            # Load data
            df = pd.read_csv(os.path.join(inputdir,fname))
            if ch == 'ch2':
                df2 = df[['imagefile','pred']]
                df2 = df2.rename(columns={'pred': ch+ch_marker_dict[ch]+'pred'})
                dfcombine = pd.concat([dfcombine, df2], axis=1)
            else:
                df2 = df[['pred']]
                df2 = df2.rename(columns={'pred': ch+ch_marker_dict[ch]+'pred'})
                dfcombine = pd.concat([dfcombine, df2], axis=1)
        
        # Modify dfcombine
        dfcombine = dfcombine.replace({'imagefile': {'_ch_2.png': ''}}, regex=True)
        
        # Output dfcombine
        outname = os.path.basename(inputdir)+'_'+w+'_combch.csv'
        outpath = os.path.join(outdir,outname)
        dfcombine.to_csv(outpath)
        print('complete '+ w)
    # End of function

    def main(self):
        # Main directory
        inputdir = r'C:\Users\Input'
        outdir = r'C:\Users\Output'
        # """Combine all"""
        combine_foldch_allpatients_for_heatm(inputdir,outdir)

c = Combine()
# Main directory
inputdir = r'C:\Users\Input'
outdir = r'C:\Users\Output'
c.combine_ncsv_to_1csv_colstack(inputdir,outdir)
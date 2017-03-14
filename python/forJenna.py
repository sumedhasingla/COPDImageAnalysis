
# ======= bunch of helper functions

def  plotLowDimWithColor( lowdim, allDataTable, colName, validSubjList, 
                         visualizer='matplotlib', ax=None, plotTitle=None, markerSize=2 ):
    """
    plot low dimensional embedding
    """

    # which subjects are mot the table
    inFeature = lowdim
    removeRowIdx = np.zeros((len(inFeature),)) ;
    for idx, s in enumerate(validSubjList):
        if not(any(allDataTable.sid.isin([s]))):
            removeRowIdx[idx] = 1

    # make an array for for the color of the subject        
    subjColor = np.zeros((len(inFeature),))
    for  idx, s in enumerate(validSubjList):
        if (removeRowIdx[idx]==0):
            c = (allDataTable[colName][ allDataTable.sid == s ])
            subjColor[idx] = c.iat[0]

    # remove invalid subjects        
    inFeature = inFeature[np.nonzero(removeRowIdx==0)]
    subjColor = subjColor[np.nonzero(removeRowIdx==0)]

    # visualize 
    if visualizer=='matplotlib':
        maxVal = -np.sort(-subjColor)[0]
        minVal = np.sort(subjColor)[0]
        cm = plt.cm.get_cmap('RdYlBu')
        if ax==None:
            scatter(inFeature[:,0], inFeature[:,1], c=subjColor, 
                    lw = 0, vmin=minVal, vmax=maxVal,  
                    cmap=cm, s=markerSize )
            colorbar()
            plt.xlim(np.percentile(inFeature[:,0], 1, axis=0), 
                     np.percentile(inFeature[:,0], 99, axis=0))
            plt.ylim(np.percentile(inFeature[:,1], 1, axis=0), 
                     np.percentile(inFeature[:,1], 99, axis=0))   
            plt.title(plotTitle)      
        else:
            scPlot = ax.scatter(inFeature[:,0], inFeature[:,1], c=subjColor, 
                    lw = 0, vmin=minVal, vmax=maxVal,  
                    cmap=cm, s=markerSize  )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(scPlot, cax=cax)
            ax.set_xlim(np.percentile(inFeature[:,0], 1, axis=0), 
                     np.percentile(inFeature[:,0], 99, axis=0))
            ax.set_ylim(np.percentile(inFeature[:,1], 1, axis=0), 
                     np.percentile(inFeature[:,1], 99, axis=0)) 
            ax.set_title(plotTitle)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
    elif visualizer=='bokeh':
        import bokeh.plotting as bokehPlotting
        from bokeh.io import output_notebook
        
        TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,tap,previewsave,box_select,poly_select,lasso_select"
        #bokehPlotting.output_file("color_scatter.html", title="color_scatter.py example")
        output_notebook()
        maxVal = -np.sort(-subjColor)[0]
        minVal = np.sort(subjColor)[0]
        cm = plt.cm.get_cmap('RdYlBu')
        
        p = bokehPlotting.figure(tools=TOOLS)
        p.scatter(inFeature[:,0], inFeature[:,1], fill_color=subjColor, fill_alpha=0.6, line_color=None, vmin=minVal, vmax=maxVal,  cmap=cm )
            
        bokehPlotting.show(p)


def evalFeatures_Regression(XTrain,yTrain,XTest,yTest,m):
    """
    evaluate a regression model
    """
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler

    # scaling
    stdScaler = StandardScaler()
    stdScaler.fit(XTrain)
    XTrain = stdScaler.transform(XTrain)
    yTrain = np.log10(yTrain)
    XTest = stdScaler.transform(XTest)
    yTest = np.log10(yTest)
        
    m.fit(XTrain, yTrain)
    pred_y = m.predict(XTest)
    r2 = metrics.r2_score(yTest, pred_y)
    s = metrics.mean_squared_error(yTest, pred_y)
    return r2, s





# ============== cholesky
# K_PSD is the result of projection
regParam =  1e-3
hidim = np.linalg.cholesky( K_PSD+\
                           regParam * np.eye( K_PSD.shape[0] )  )



# Numerical checking
print "This is some information the projected kernel :"
print "min, median, max value of the kernel: ",np.min(divKernel_PSD), np.median(divKernel_PSD), np.max(divKernel_PSD)
print "less than 0.3 :", np.sum(divKernel_PSD.flatten() < 0.3)/np.float(np.prod(divKernel_PSD.shape))
print "less than 0.5 :", np.sum(divKernel_PSD.flatten() < 0.5)/np.float(np.prod(divKernel_PSD.shape))
print "less than 1.0 :", np.sum(divKernel_PSD.flatten() < 1)/np.float(np.prod(divKernel_PSD.shape))
print "-----------"
print "number of subject with at least one entry higher than 1: ", ((divKernel_PSD > 1.0).sum(axis=0)>0).sum()
print "These are diagonal elements: ", np.diag(divKernel_PSD)


_ = hist(divKernel_PSD.flatten()[ np.logical_and(divKernel_PSD.flatten() < 1 , divKernel_PSD.flatten() > 0) ],50 )

imshow(divKernel_PSD, vmin=0, vmax=1.0, cmap='cool')




# =================== embedding for patient
from sklearn import manifold

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
tSNE = tsne.fit_transform(hidim)

#clf = LocallyLinearEmbedding(n_components=d)
#tSNE = clf.fit_transform(hidim)    

# plot 2d embedding
import seaborn as sns
permIdx = np.random.permutation(range(len(subjList)))
subjList_subset = np.array(subjList)[permIdx[:3000]]    
with sns.axes_style("white"):
    f, ax1 = plt.subplots()
    
tSNE_subset = tSNE[permIdx[:3000]]

# write your own visualizationb, each subject is a dot in 2D and color each subject with her own y
plotLowDimWithColor(tSNE_subset, phenotypeDB_clean, 'FEV1pp_utah' , subjList_subset, 
                    ax=ax1, plotTitle='tSNE', markerSize=20)

ax1.set_xlim([-11,11])
ax1.set_ylim([-11,11])
ax1.set_title('Patient Embedding ($d=2$)',fontsize=20)


# ========== extracting higher dim features per subject
from sklearn.manifold import LocallyLinearEmbedding

clf = LocallyLinearEmbedding(n_components=100)
LLE_modified = clf.fit_transform(hidim)   # this is your X, ie your feature per subject


# ======== Regression
numPCA_comp = 100
colNameList = []
for numDim in range(2,numPCA_comp,10):
    colNameList.append(['LLE_modified_f%d'%i for i in range(numDim)])



score_r2DictList = {}
p_r2DictList = {}
accScoreDictList = {}
p_accDictList = {}    
    

featureName = 'histFHOG_largeRange_setting1'
score_r2DictList[featureName] = []
p_r2DictList[featureName] = []
accScoreDictList[featureName] = []
p_accDictList[featureName] = []
for colName in colNameList:
    #---------------------  Compute BOW
    tic()
    print "computing bag of words ......"
    bags = [ d['I'] for d in data ]     # here the data is your patient patch data
    feats = Features(bags)
    bow = BagOfWords(KMeans(n_clusters=len(colName), max_iter=100, n_init=2))
    bowized = bow.fit_transform(feats)
    print "Done !"
    toc()

    # make dataframe for the LLE features
    LLE_modified_df = pd.DataFrame(LLE_modified[:,0:len(colName)],
                                   columns=['LLE_modified_f%d'%i for i in range(len(colName))])
    LLE_modified_df['sid'] = subjList

    # make dataframe out of BOW
    BOW_df = pd.DataFrame(bowized,columns=['BOW_f%d'%i for i in range(bowized.shape[1])])
    BOW_df['sid'] = subjList    

    # merge two dataframes
    complete_db = pd.merge(phenotypeDB_clean, LLE_modified_df, left_on='sid',right_on='sid')
    complete_db = pd.merge(complete_db, BOW_df, left_on='sid',right_on='sid')
    complete_db_clean = complete_db.dropna(subset=['Insp_Below950_Slicer','Exp_Below950_Slicer'])


    
    
    # make a cross-validation object
    cv = KFold(n=NUMBER-OF-SUBJECTS, 
               n_folds=50, shuffle=False, random_state=0)

 
    ### regression
    allXy = {}    
    yName = 'FEV1pp_utah'

    # the output of getXyPair is tuple of X,y
    #allXy['LLE_modified'] = getXyPair(complete_db_clean,\
    #                   colName,\
    #                   yName, verbose=0)    
    #allXy['BOW'] = getXyPair(complete_db_clean,\
    #               [col for col in complete_db_clean.columns if 'BOW_f' in col],\
    #               yName, verbose=0) 
    #allXy['classic'] = getXyPair(complete_db_clean,\
    #                    [col for col in complete_db_clean.columns if '950' in col],\
    #                    yName, verbose=0)
    allXy['JennaFeatues'] = (X,y)
    allXy['BOW'] = (bowized,y)


    score_r2Dict = {}
    p_r2Dict = {}
    for k in allXy.keys():  
        from sklearn import linear_model
        
        #clf = linear_model.Ridge(alpha=1)
        score_r2Dict[k] = [] 
        #score_r2Dict[k] = cross_val_score(clf, allXy[k][0], allXy[k][1], cv=50, scoring='r2_score')
        print "working on ",k, "...."
        for i, (train, test) in enumerate(cv):
            clf = linear_model.Ridge(alpha=1)
            r2,s = evalFeatures_Regression(allXy[k][0][train],    # your X (LLE_modified)
                                           allXy[k][1][train],    # your y 
                                           allXy[k][0][test],     # your X
                                           allXy[k][1][test],clf) # your y
            score_r2Dict[k].append(r2)


    print "****** R2 ********* : dim : ", len(colName) 
    for k in allXy.keys():
        _,p = scipy.stats.ttest_rel(np.array(score_r2Dict[k]), 
                              np.array(score_r2Dict['classic']))              
        p_r2Dict[k] = p
        print('MEAN R2 for %s (Ridge Reg):    %f (%f) -- pVal: %.3e'%\
              (k,np.mean(score_r2Dict[k]), np.std(score_r2Dict[k]),p) )  

    # add the results to a list
    score_r2DictList[featureName].append(score_r2Dict)
    p_r2DictList[featureName].append(p_r2Dict)
    accScoreDictList[featureName].append(accScoreDict)
    p_accDictList[featureName].append(p_accDict)


pk.dump( score_r2DictList, open(hellingerRoot + "/linReg_score_r2DictList.p",'wb') )
pk.dump( p_r2DictList, open(hellingerRoot + "/linReg_p_r2DictList.p",'wb') )
pk.dump( accScoreDictList, open(hellingerRoot + "/linReg_accScoreDictList.p",'wb') )
pk.dump( p_accDictList, open(hellingerRoot + "/linReg_p_accDictList.p",'wb') )






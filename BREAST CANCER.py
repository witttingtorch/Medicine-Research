import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
Breast_cancer =pd.read_csv("data.csv")
Breast_cancer.head()
Breast_cancer.columns
Breast_cancer =pd.read_csv("data.csv")
Breast_cancer.head()
Breast_cancer.info()
Breast_cancer.isnull().any()
del(Breast_cancer['Unnamed: 32'])
Breast_cancer.info()
cols=Breast_cancer.columns
cmap=sns.diverging_palette(230,10,as_cmap=True)
mask=np.zeros_like(Breast_cancer[cols].corr(),dtype= np.bool)
mask[np.triu_indices_from(mask)]=True
f, ax = plt.subplots(figsize=(25, 10))
ax = sns.heatmap(Breast_cancer[cols].corr(),mask=mask,cmap=sns.cubehelix_palette(100,light=.95,dark=.15),vmax=.3,center=0,annot=True,
               square =True, linewidths=.5, cbar_kws={"shrink":.3})
x=Breast_cancer.iloc[:,2:].values
y=Breast_cancer.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
label_encoder.classes_
label_encoder.transform(['M','B'])
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y,random_state=1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
piping=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
piping.fit(x_train,y_train)
predict_labels=piping.predict(x_test)
print('Test accuracy:%3f'%piping.score(x_test,y_test))
from sklearn.model_selection import StratifiedKFold
k_fold_no=StratifiedKFold(n_splits=10,random_state=1).split(x_train,y_train)
scores=[]
for k,(train,test) in enumerate(k_fold_no):
    piping.fit(x_train[train],y_train[train])
    score=piping.score(x_train[test],y_train[test])
    scores.append(score)

print('Fold: %2d,Class dist:%s,Acc:%.3f' % np.bincount(k+1,np.bincount(y_train['train']),score))
print('\ncross validations Accuracy:%.3f +-%3f' %np.mean(scores),np.std(scores) )
from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=piping,x=x_train,y=y_train,cv=10,n_jobs=1)
print('cross validation scores: %s'% scores)
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
piped_up=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',random_state=1))
train_sizes,train_scores,test_scores=learning_curve(estimator=piped_up,x=x_train,y=y_train,train_to_loss function.sizes=np.linspace(0.1,1.0,10)CV=10,n_jobs=1)
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',markersize=5,label='Training Accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',makersize=5,label='Validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import validation_curve
parameter_range=[0.001,0.01,0.1,1.0,10.0,100.0]
train_scores,test_scores = validation_curve(estimator=piped_up,x=x_train,y=y_train,param_name='Logisticregression_C',param_range=parameter_range,cv=10)
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
plt.plot(parameter_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(parameter_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color=blue)
plt.plot(parameter_range,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(parameter_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('parameter c')
plt.ylabel('Accuracy')
plt.ylim()
plt.show()
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_support_vector=make_pipeline(StandardScaler(),SVC(random_state=1))
parameter_range=[0.001,0.01,0.1,1.0,10.0,100.0]
parameter_grid=[{'SVC__C':parameter_range,'svc__kernel':['linear']},{'svc__c':parameter_range,'svc__gamma':parameter_range,'svc__kernel':['RBF']}]
grid_search=GridSearchCV(estimator=pipe_support_vector,param_grid=parameter_grid,scoring='accuracy',cv=10,n_jobs=1)
grid_search=grid_search.fit(x_train,y_train)
print('Best Score:',grid_search.best_score_)
print('Best_parameters :',grid_search.best_params_)
classify=grid_search.best_estimator_
classify.fit(x_train,y_train)
print('Test accuracy :%.3f' %classify.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
grid_s=GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid=[{ 'Max Depth':[1,2,3,4,5,6,7,None]}],scoring='accuracy',cv=2)
Scores=cross_val_score(grid_s,x_train,y_train,scoring='Accuracy',cv=5)
print('Cross validation accuracy:%.3f +-%.3f'%(np.mean(Scores),np.std(Scores)))
from sklearn.metrics import confusion_matrix
pipe_support_vector.fit(x_train,y_train)
y_prediction=pipe_support_vector.predict(x_test)
confusion=confusion_matrix(y_true=y_test,y_pred=y_prediction)
print(confusion)
fig , ax= plt.subplots(figsize(2.5,2.5))
ax.matshow(confusion,cmap=plt.cm.blues,alpha=0.3)
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(x=j,y=i,s=confusion[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
print('precision %3.f:'%precision_score(y_true=y_test,y_pred=y_prediction))
print('Recall%.3f'% recall_score(y_true=y_test,y_pred=y_prediction))
print('f1 score %.3f'%f1_score(y_true=y_test,y_pred=y_prediction))
from sklearn.metrics import make_scorer,f1_score
score=make_scorer(f1_score,pos_label=0)
grad_s=GridSearchCV(estimator=pipe_support_vector,param_grid=parameter_grid,scoring=score,cv=10)
grad_s=grad_s.fit(x_train,y_train)
print(grad_s.best_score_)
print(grad_s.best_params_)
from sklearn.metrics import roc_curve,auc
from scipy import interpolate
piping_sys=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(penalty='l2',random_state=1,C=100.0))
x_train2=x_train[:,[4,14]]
cross_values=list(StratifiedKFold(n_splits=3,random_state=1).split(x_train,y_train))
plot=plt.figure(figsize=(7,5))
mean_true_positive_rate=0.0
mean_false_positive_rate=np.linspace(0,1,100)
all_true_positive=[]
for i ,(train,test) in enumerate(cross_values):
    probability=piping_sys.fit(x_train2[train],y_train[train]).predict_proba(x_train2[test])
    fpr,tpr,threshold=roc_curve(y_train[test],probability[:,1],pos_label=1)
mean_true_positive_rate+=interpolate(mean_false_positive_rate,fpr,tpr)
mean_true_positive_rate[0]=0.0
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,label='ROC fold % d(area=%0.2f)0'%(i+1,roc_auc))
plt.plot([0,1],[0,1],linestyle=':',color=(0.6,0.6,0.6),label='random_guessing')
mean_true_positive_rate /=len(cross_values)
mean_true_positive_rate[-1]=1.0
mean_auc=auc(mean_false_positive_rate,mean_true_positive_rate)
plt.plot(mean_false_positive_rate,mean_true_positive_rate,'k--',label='mean ROC(Area %0.2f)'%mean_auc,lw=2)
plt.plot([0,0,1],[0,1,1],linestyle=':',color='black',label='perfect petrfomance')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.show()
pre_score=make_scorer(score_func=precision_score,pos_label=1,greater_is_better=True,average='micro')
x_imbalance=np.vstack((x[y==0],x[y==1][:40]))
y_imbalance=np.hstack((y[y==0],y[y==1][:40]))
y_predict=np.zeros(y_imbalance.shape[0])
np.mean(y_predict=y_imbalance)*100
from sklearn.utils import resample
print('Number of class 1 samples before',x_imbalance[y_imbalance==1].shape[0])
x_upsampled,y_upsampled=resample(x_imbalance[y_imbalance==1],y_imbalance[y_imbalance==1],replace=True,n_samples=x_imbalance[y_imbalance==0].shape[0],random_state=123)
print('Number of class 1 sample after:',x_upsample.shape[0])
x_balance=np.vstack((x[y==0],x_upsampled))
y_balance=np.hstack((y[y==0],y_upsampled))
y_predicted=np.zeros(y_balance.shape[0])
np.mean(y_predicted=y_balance)*100



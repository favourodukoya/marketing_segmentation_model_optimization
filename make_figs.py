import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

OUT='outputs'
d = np.load(f'{OUT}/data_cache.npz'); y_test = d['y_test']
res = pd.DataFrame(json.load(open(f'{OUT}/results.json')))
opt_order = ['sgd','momentum','rmsprop','adam','nadam']
palette = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974']

def mean_metric(name,k):
    return res[res['name']==name][k].mean()

# FIG 1: grouped bar of metrics per optimizer
metrics=['accuracy','precision','recall','f1','roc_auc']; labels=['Accuracy','Precision','Recall','F1','ROC AUC']
x=np.arange(len(opt_order)); width=0.16
fig,ax=plt.subplots(figsize=(12,6))
for i,(m,lab) in enumerate(zip(metrics,labels)):
    vals=[mean_metric(n,m) for n in opt_order]
    ax.bar(x+(i-2)*width,vals,width,label=lab,color=palette[i])
ax.set_xticks(x); ax.set_xticklabels([o.upper() for o in opt_order],fontsize=10)
ax.set_ylabel('Score (mean over seeds)',fontsize=11); ax.set_ylim([0,1.0])
ax.set_title('Test Metrics by Optimizer (class weighted, fixed architecture)',fontsize=13,pad=12)
ax.legend(ncol=5,fontsize=9,loc='lower center',bbox_to_anchor=(0.5,-0.16)); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/optimizer_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print('fig1 optimizer_comparison done')

# FIG 2: validation loss curves (seed 42)
fig,ax=plt.subplots(figsize=(10,6))
for n,c in zip(opt_order,palette):
    vl=json.load(open(f'{OUT}/valloss_{n}.json'))
    ax.plot(range(1,len(vl)+1),vl,label=n.upper(),linewidth=2,color=c)
ax.set_xlabel('Epoch',fontsize=11); ax.set_ylabel('Validation Loss',fontsize=11)
ax.set_title('Validation Loss per Epoch by Optimizer (seed 42)',fontsize=13,pad=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/loss_curves.png',dpi=150,bbox_inches='tight'); plt.close()
print('fig2 loss_curves done')

# FIG 3: ROC curves (seed 42 probas)
fig,ax=plt.subplots(figsize=(8,7))
for n,c in zip(opt_order,palette):
    p=np.load(f'{OUT}/proba_{n}.npy'); fpr,tpr,_=roc_curve(y_test,p)
    ax.plot(fpr,tpr,label=f'{n.upper()} (AUC={roc_auc_score(y_test,p):.3f})',linewidth=2,color=c)
ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Random')
ax.set_xlabel('False Positive Rate',fontsize=11); ax.set_ylabel('True Positive Rate',fontsize=11)
ax.set_title('ROC Curves by Optimizer (Test Set, seed 42)',fontsize=13,pad=12)
ax.legend(fontsize=10,loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/roc_curves.png',dpi=150,bbox_inches='tight'); plt.close()
print('fig3 roc_curves done')

# FIG 4: confusion matrix for adam (best), seed 42
p=np.load(f'{OUT}/proba_adam.npy'); yp=(p>=0.5).astype(int)
cm=confusion_matrix(y_test,yp)
fig,ax=plt.subplots(figsize=(7,5.5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Retain','Churn'],yticklabels=['Retain','Churn'],
            linewidths=0.5,linecolor='white',ax=ax,annot_kws={'size':14})
ax.set_title('Confusion Matrix: Adam (Test Set)',fontsize=12,pad=12)
ax.set_xlabel('Predicted',fontsize=11); ax.set_ylabel('Actual',fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/confusion_matrix.png',dpi=150,bbox_inches='tight'); plt.close()
print('fig4 confusion done')

# FIG 5: baseline vs best (adam) before/after on key metrics
keys=['accuracy','precision','recall','f1','roc_auc']; klab=['Accuracy','Precision','Recall','F1','ROC AUC']
base=[mean_metric('baseline',k) for k in keys]
best=[mean_metric('adam',k) for k in keys]
x=np.arange(len(keys)); width=0.36
fig,ax=plt.subplots(figsize=(10,6))
b1=ax.bar(x-width/2,base,width,label='Existing model (SGD, no weights)',color='#B0B0B0')
b2=ax.bar(x+width/2,best,width,label='Optimized (Adam + class weights)',color='#4C72B0')
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{b.get_height():.2f}',ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(klab,fontsize=10); ax.set_ylim([0,1.0])
ax.set_ylabel('Score',fontsize=11)
ax.set_title('Existing Model vs Optimized Model',fontsize=13,pad=12)
ax.legend(fontsize=10,loc='upper right'); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/before_after.png',dpi=150,bbox_inches='tight'); plt.close()
print('fig5 before_after done')

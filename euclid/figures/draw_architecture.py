"""Render the current bright-sample frozen-encoder CLIP architecture."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(16,11))
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02)
ax.set(xlim=(0,16),ylim=(0,11));ax.axis('off')
ink='#172B42'; muted='#52657A'; blue='#22669A'; teal='#007F78'; orange='#BA681D'
def text(x,y,s,size=11,color=ink,weight='normal',ha='left',va='center'):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.25)
def box(x,y,w,h,title,body='',face='#F1F6FA',edge='#CBD7E2',titlecolor=ink):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.12',
                              facecolor=face,edgecolor=edge,linewidth=1.15))
    text(x+.18,y+h-.27,title,11,titlecolor,'bold')
    if body:text(x+.18,y+h-.48,body,9.3,va='top')
def arrow(x,y,xx,yy,color=muted,style='-'):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=13,
                                 linewidth=1.4,color=color,linestyle=style))
text(.2,10.65,'Euclid images × star-formation histories',23,weight='bold')
text(.2,10.22,'Pretrained representations → frozen encoders → trainable MLP alignment',13,muted)
text(.2,9.68,'01   PRETRAIN REPRESENTATIONS',12,blue,'bold')
# Two clearly separate pretraining pathways.
box(.2,6.82,4.7,2.48,'Image representation · pretrained ZooBot',
    'ZooBot Euclid morphology encoder\nImported pretrained weights from Hugging Face\n\nmwalmsley/zoobot-encoder-euclid',face='#F0F5FA')
text(.4,7.07,'Retain the image backbone; discard classification outputs.',9,muted)
box(5.15,6.82,10.6,2.48,'SFH representation · masked autoencoder pretraining',face='#F0F8F6',edge='#BBDCD5',titlecolor=teal)
box(5.38,7.58,2.30,.98,'Posterior SFH draw','250 bins · log weights\nMask 35% contiguous bins',face='white')
box(8.0,7.58,2.07,.98,'Transformer encoder','4 layers · 4 heads\nwidth 128 → latent 256',face='white')
box(10.40,7.58,2.14,.98,'Transformer decoder','2 layers · 4 heads\n250-bin softmax output',face='white')
box(12.87,7.58,2.61,.98,'Reconstruct median SFH','0.5 × weighted Huber\n+ 0.5 × cumulative W₁',face='white')
for x,xx in [(7.70,7.98),(10.09,10.38),(12.56,12.85)]:arrow(x,8.04,xx,8.04,teal)
text(5.4,7.15,'Encoder + decoder trained here. Only the encoder is transferred to the alignment stage.',10,teal)
# Transfer arrows to stage boundary, explicit text avoids cross-branch lines.
arrow(2.5,6.80,2.5,6.43,blue,'--')
arrow(9.04,6.80,9.04,6.43,teal,'--')
text(.2,6.12,'02   ALIGN MATCHED IMAGE–SFH PAIRS',12,blue,'bold')
text(15.65,6.12,'Bright sample: VIS < 22 · 136,983 paired galaxies',10,muted,ha='right')
# Alignment rows
for y,label in [(4.65,'IMAGE'),(2.75,'SFH')]:text(.25,y+1.18,label,10,blue if label=='IMAGE' else teal,'bold')
box(.2,4.65,3.0,.90,'Euclid VIS stamp','224 × 224 · size-scaled crop\nGrayscale replicated to 3 channels')
box(3.60,4.65,3.10,.90,'ZooBot image backbone','FROZEN · inference mode\nPretrained Euclid features',face='#EAF2FA',edge=blue,titlecolor=blue)
box(7.10,4.65,3.40,.90,'Image MLP adapter','TRAINABLE · d_image → 256 → 256\nGELU + dropout',face='#FFF3E6',edge=orange,titlecolor=orange)
box(10.90,4.65,1.65,.90,'L₂ normalize','256-D image\nembedding',face='#F3EFF9')
box(.2,2.75,3.0,1.02,'Median SFH shape','250 common fractional-time bins\nUnit-sum weights → log₁₀\nMedian only; no posterior sampling')
box(3.60,2.75,3.10,1.02,'SFH transformer encoder','FROZEN · pretrained autoencoder\nTime/value tokens + positions\nCLS pooling → 256-D latent',face='#EAF5F2',edge=teal,titlecolor=teal)
box(7.10,2.75,3.40,1.02,'Residual SFH MLP adapter','TRAINABLE · norm → 512 → 256\nGELU; residual branch starts at zero\nh′ = h + 0.1 × MLP(h)',face='#FFF3E6',edge=orange,titlecolor=orange)
box(10.90,2.75,1.65,1.02,'L₂ normalize','256-D SFH\nembedding',face='#F3EFF9')
for y in [5.07,3.24]:
    for x,xx in [(3.22,3.58),(6.72,7.08),(10.52,10.88)]:arrow(x,y,xx,y)
box(13.0,3.02,2.75,2.50,'Contrastive alignment','Cosine similarities / temperature\n\nSymmetric image ↔ SFH\ncross-entropy loss\n\nSame galaxy = positive pair\nOther batch galaxies = negatives',face='#F3EFF9',edge='#8C75AC')
arrow(12.57,5.06,12.98,4.98,'#80649B')
arrow(12.57,3.25,12.98,3.5,'#80649B')
text(.25,2.31,'TRAINED IN THIS STAGE',9,orange,'bold')
text(3.0,2.31,'Two MLP adapters + temperature only. Batch size 128; no queue, soft positives, or reconstruction loss.',10,muted)
# downstream distinct strip
text(.2,1.77,'03   USE THE ALIGNED EMBEDDINGS',12,blue,'bold')
box(.2,.47,7.45,1.02,'Scientific exploration','UMAP · image/SFH retrieval · morphology and SFH diagnostics\nPhysical-property and MS overlays are analysis tools, not training inputs.',face='#F7F9FB')
box(8.0,.47,7.75,1.02,'Separate downstream experiment · conditional diffusion','Frozen aligned SFH embedding conditions a pixel-space diffusion U-Net\nGenerate 224 × 224 VIS images; shared noise seeds enable condition comparisons.',face='#F7F9FB')
text(.25,.14,'Current configuration: training_bright_frozen_mlp  •  SFH pretraining: sfh_autoencoder_v1',8.5,muted)
text(15.65,.14,'Blue / teal: frozen during alignment     Orange: trainable adapters',8.5,muted,ha='right')
out=Path(__file__).parent/'euclid_current_architecture'
for ext in ['png','pdf','svg']:
    fig.savefig(out.with_suffix('.'+ext),dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.15)
print(out)

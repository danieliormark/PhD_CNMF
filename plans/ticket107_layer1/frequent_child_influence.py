# child×article: total column influence (sum of squared cells, log(1+tf)·idf) of the most frequent child hyperedges
import math,collections,os
os.chdir("/mnt/hum01-home01/p91688di/PhD_CNMF/plans/ticket107_layer1")
exec(open("layer1_review.py").read().split("sq = collections.Counter()")[0])
top=sorted(arts,key=lambda k:-len(arts[k]))[:8]
tot=collections.Counter()
for (k,a),t in tf.items(): tot[k]+=(math.log1p(t)*idf_a[k])**2
near=[k for k in arts if 700<=len(arts[k])<=900]
print("N",N)
for k in top+near: print(sorted(k),len(arts[k]),round(idf_a[k],2),round(tot[k]))

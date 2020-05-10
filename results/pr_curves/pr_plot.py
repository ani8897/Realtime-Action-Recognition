import matplotlib.pyplot as plt

lrcn_precision = [0.48841355, 0.89473684, 1.] 
lrcn_recall = [1., 0.99270073, 0.] 
c3d_precision = [0.5258467, 0.95666667, 1.] 
c3d_recall = [1., 0.97288136, 0.]
tsm_precision = [0.50089127, 0.9751773, 1.] 
tsm_recall = [1., 0.97864769, 0.]

fig, ax=plt.subplots()
ax.step(lrcn_recall, lrcn_precision,color='b',alpha=0.99,where="post",label="LRCN: 0.892")
ax.fill_between(lrcn_recall, lrcn_precision, alpha=0.2, color='b', step="post")
ax.step(c3d_recall, c3d_precision,color='r',alpha=0.99,where="post",label="C3D: 0.944")
ax.fill_between(c3d_recall, c3d_precision, alpha=0.2, color='r', step="post")
ax.step(tsm_recall, tsm_precision,color='g',alpha=0.99,where="post",label="TSM: 0.965")
ax.fill_between(tsm_recall, tsm_precision, alpha=0.2, color='g', step="post")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend()
# plt.title("2-class Precision-Recall curve: AP={0:0.2f}".format(average_precision))
plt.savefig('pr.png', dpi=600)
plt.close(fig)
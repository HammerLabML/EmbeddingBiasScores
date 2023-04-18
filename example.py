import pickle
from embedding import BertHuggingface
import math
from geometrical_bias import SAME, DirectBias, WEAT, RIPA, MAC, GeneralizedWEAT
import numpy as np
from lipstick import BiasGroupTest, NeighborTest, ClusterTest, ClassificationTest


jobs = ['nurse', 'doctor', 'teacher', 'police officer', 'firefighter', 'secretary', 'programmer', 'engineer', 'caretaker', 'salesclerk']
jobs_m = ['doctor', 'police officer', 'firefighter', 'programmer', 'engineer', 'surgeon', 'rapper', 'businessman', 'pastor']
jobs_f = ['nurse', 'teacher', 'secetrary', 'caretaker', 'salesclerk', 'model', 'paralegal', 'dietitian', 'teacher']

jobs_black = ['taxi driver', 'basketball player']
jobs_white = ['police officer', 'lawyer']
jobs_asian = ['programmer', 'mathematician']

gender_attributes = [['he', 'man', 'his', 'boy', 'son', 'himself', 'father'], ['she', 'woman', 'her', 'girl', 'daughter', 'herself', 'mother']]
race_attributes = [['black', 'african'], ['white', 'caucasian'], ['asian', 'chinese']]


bert = BertHuggingface(2)

job_emb = bert.embed(jobs)
job_m_emb = bert.embed(jobs_m)
job_f_emb = bert.embed(jobs_f)
jobs_black_emb = bert.embed(jobs_black)
jobs_white_emb = bert.embed(jobs_white)
jobs_asian_emb = bert.embed(jobs_asian)
gender_attr = [bert.embed(attr) for attr in gender_attributes]
race_attr = [bert.embed(attr) for attr in race_attributes]


gweat = GeneralizedWEAT()
gweat.define_bias_space(gender_attr)

gweat2 = GeneralizedWEAT()
gweat2.define_bias_space(race_attr)

mac = MAC()
mac.define_bias_space(gender_attr)

weat = WEAT()
weat.define_bias_space(gender_attr)

same = SAME()
same.define_bias_space(gender_attr)

db1 = DirectBias(k=1,c=1)
db1.define_bias_space(gender_attr)

score_names = ['mac', 'db1', 'same', 'weat']
scores = [mac, db1, same, weat]

for i in range(len(scores)):
    print(score_names[i], ": ", [scores[i].individual_bias(emb) for emb in job_emb])

# most scores implement a mean bias
for i in range(len(scores)-1):
    print(score_names[i], ": ", scores[i].mean_individual_bias(job_emb))
    
# weat implements an effect size over two groups stereotypically associated with the gender attribute groups
print("weat: ", weat.group_bias([job_m_emb, job_f_emb]))
print("gweat (gender): ", gweat.group_bias([job_m_emb, job_f_emb]))
print("gweat (race): ", gweat2.group_bias([jobs_black_emb, jobs_white_emb, jobs_asian_emb]))


same.define_bias_space(race_attr)
print("Black vs. White")
print("Skew: ", same.skew_pairwise(job_emb, 0, 1))
print("Stereotype: ", same.stereotype_pairwise(job_emb, 0, 1))
print()

print("Asian vs. White")
print("Skew: ", same.skew_pairwise(job_emb, 2, 1))
print("Stereotype: ", same.stereotype_pairwise(job_emb, 2, 1))
print()

same.define_bias_space(race_attr)
print("Multiclass bias vector for 'nurse': ", same.individual_bias_per_pair(job_emb[0])) # first is black/white, second black/asian
print("bias magntiude for 'nurse': ", same.individual_bias(job_emb[0]))


neighborTest = NeighborTest(k=5)

# this is how the neighbor test is used in the paper:
# TODO: call weat on jobs, sort by bias into m/f groups
weats = [weat.individual_bias(emb) for emb in job_emb]
sort_idx = np.argsort(weats)
jobs_f_weat = [job_emb[idx] for idx in sort_idx[:5]]
jobs_m_weat = [job_emb[idx] for idx in sort_idx[-5:]]
print("bias by neighbor (as in the paper):")
print(neighborTest.bias_by_neighbor([jobs_f_weat, jobs_m_weat]))

# instead of using weat we can define stereotypical groups by hand
jobs_gender = [job_m_emb, job_f_emb]
print("bias by neighbor (without weat):")
print(neighborTest.bias_by_neighbor(jobs_gender))

# define the bias space with a subset of known stereotypical words, then test words without known categories
neighborTest.define_bias_space(gender_attr)
print("bias by neighbor (without predefined groups): ")
biases = [neighborTest.individual_bias(emb) for emb in job_emb]
print(biases)

print("mean bias by neighbor (without predefined groups): ")
print(neighborTest.mean_individual_bias(job_emb))

clusterTest = ClusterTest()

# according to the paper
weats = [weat.individual_bias(emb) for emb in job_emb]
sort_idx = np.argsort(weats)
jobs_f_weat = [job_emb[idx] for idx in sort_idx[:5]]
jobs_m_weat = [job_emb[idx] for idx in sort_idx[-5:]]
print("cluster test accuracy (weat): ")
print(clusterTest.cluster_test([jobs_f_weat, jobs_m_weat]))

# instead of using weat we can define stereotypical groups by hand
jobs_gender = [job_m_emb, job_f_emb]
print("cluster test accuracy (predefined groups): ")
print(clusterTest.cluster_test(jobs_gender))

# define the bias space with a subset of known stereotypical words, then test words without known categories
#clusterTest.define_bias_space(gender_attr)
#clusterTest.mean_individual_bias(job_emb)


clfTest = ClassificationTest()
cv_scores = clfTest.classification_test(jobs_gender)
print(np.mean(cv_scores), np.std(cv_scores))
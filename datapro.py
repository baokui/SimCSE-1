#############################################
import sys
from modules import getfiles, readData1
# train data
path_data = '/search/odin/guobk/data/Tab3/'
files = getfiles(path_data)
#files = [f for f in files if '0501' not in f]
S = readData1(files,adtypes=['sear','sema'])
#S1 = [s for s in S if len(s[1].strip())>=5 and s[4][:4]!='mult' and s[4][:4]!='godT' and s[4][:4]!='poet']
Docs = getDocs()
for k in Docs:
    Docs[k] = Docs[k].replace('\n',' ').replace('\r',' ').replace('\t',' ')
S1 = S
R1 = [] # search & semanticSearch
for i in range(len(S1)):
    if i%10000==0:
        print(i,len(S1),len(R1))
    docs = S1[i][3].split('#')
    adtypes = S1[i][4].split('#')
    clicks = S1[i][2].split('#')
    if adtypes and adtypes[0] not in ['search','semanticSearch']:
        continue
    contents = [Docs[d] for d in docs if d in Docs]
    contents_c = [Docs[d] for d in clicks if d in Docs]
    R1.append([S1[i][1],contents_c,contents])
S = [[r[0].replace('\n',' ').replace('\r',' ').replace('\t',' '),r[1]] for r in R1]
T = [Docs[k] for k in Docs]
import random
epochs = 3
X = []
for epoch in range(epochs):
    for i in range(len(S)):
        if i%10000==0:
            print(epoch,i,len(S),len(X))
        q = S[i][0]
        d_pos = S[i][1]
        X.extend([[q,t,'1'] for t in d_pos])
#############################################
# test data
from modules import getfiles, readData1,getDocs
path_data = '/search/odin/guobk/data/Tab3_test/20210623'
files = getfiles(path_data)
S = readData1(files,adtypes=['sear','sema'])
#S1 = [s for s in S if len(s[1].strip())>=5 and s[4][:4]!='mult' and s[4][:4]!='godT' and s[4][:4]!='poet']
Docs = getDocs()
for k in Docs:
    Docs[k] = Docs[k].replace('\n',' ').replace('\r',' ').replace('\t',' ')
S1 = S
R1 = [] # search & semanticSearch
for i in range(len(S1)):
    if i%10000==0:
        print(i,len(S1),len(R1))
    docs = S1[i][3].split('#')
    adtypes = S1[i][4].split('#')
    clicks = S1[i][2].split('#')
    if adtypes and adtypes[0] not in ['search','semanticSearch']:
        continue
    contents = [Docs[d] for d in docs if d in Docs]
    contents_c = [Docs[d] for d in clicks if d in Docs]
    R1.append([S1[i][1],contents_c,contents])
R2 = []
for i in range(len(R1)):
    R2.append({'input':R1[i][0].strip(),'clicks':R1[i][1],'rec_ori':R1[i][2]})
with open('/search/odin/guobk/data/bert_semantic/finetuneData_new_test/Queries-test-0623.json','w') as f:
    json.dump(R2,f,ensure_ascii=False,indent=4)
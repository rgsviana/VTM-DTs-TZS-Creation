'''
	PARA EXECUTAR ESTE PROGRAMA, UTILIZE O SEGUINTE COMANDO:
		python3 random_search.py caminho_dataset n_iteracoes
	EXEMPLO:
		python3 random_search-RAMIRO.py /home/rgsviana/DATASETS_TZS/dataset_tzs_16x16_train.csv /home/rgsviana/DATASETS_TZS/dataset_tzs_16x16_test.csv 3 dataset_tzs_16x16
'''
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier as DT
from scipy.stats import pearsonr
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/home/rgsviana/DATASETS_TZS/tmp/'
args = sys.argv
if len(args)!=5:
	print("Quatro argumentos devem ser passados na chamada do programa:")
	print("\t1 - Caminho do dataset de treino a ser lido.")
	print("\t2 - Caminho do dataset de treino a ser lido.")
	print("\t3 - Numero de iteracoes para o Random Search.")
	print("\t4 - Nome do modelo associado aos resultados.")
else:
	path_train = args[1]
	path_test = args[2]
	n = int(args[3])
	model_name = args[4]

print("\nParametros de entrada:")
print("path_train ", path_train)
print("path_test ", path_test)
print("n ", n)
print("model_name ", model_name)

def largura(row):
	return int(row['bs'].split("x")[0])
def altura(row):
	return int(row['bs'].split("x")[1])

'''
df = pd.read_csv(path) #Leitura do dataset
#Separa os dados em treino e teste
df = df.sample(frac=1).reset_index(drop=True) #Este comando embaralha os exemplos
X, XT = train_test_split(df,test_size=0.25, random_state=66)
X.to_csv(model_name+"_train.csv",index=False) #Guarda em um arquivo o conjunto de treino
XT.to_csv(model_name+"_test.csv",index=False) #Guarda em um arquivo o conjunto de teste
del df
del XT
'''

X = pd.read_csv(path_train) #Leitura do dataset de treino
XT = pd.read_csv(path_test) #Leitura do dataset de teste
#X = X.sample(frac=1).reset_index(drop=True) #Este comando embaralha os exemplos
#XT = XT.sample(frac=1).reset_index(drop=True) #Este comando embaralha os exemplos
X.drop(columns=['video_width', 'video_heigh', 'bcw_index', 'cost_mv_uni_l0', 'cost_mv_uni_l1', 'cost_bi'], inplace=True)
XT.drop(columns=['video_width', 'video_heigh', 'bcw_index', 'cost_mv_uni_l0', 'cost_mv_uni_l1', 'cost_bi'], inplace=True)
X.to_csv(model_name+"_train_final.csv",index=False) #Guarda em um arquivo o conjunto de treino adicional
XT.to_csv(model_name+"_test_final.csv",index=False) #Guarda em um arquivo o conjunto de teste adicional
del XT

def classeToInt(row):
	if row['executaTZS'] == 0:
		return 0
	elif row['executaTZS'] == 1:
		return 1
	else:
		return 2
   
X['executaTZS'] = X.apply(lambda row: classeToInt(row),axis=1)
y = X['executaTZS']
X.drop(columns=['executaTZS'], inplace=True)
#X['largura'] = X.apply(lambda row : largura(row), axis=1)
#X['altura'] = X.apply(lambda row : altura(row), axis=1)
print("\nColunas:")
print(X.columns.to_list())

#Hiperparâmetros e espaço de busca
units = list(range(2,11))
tens = list(range(20,101,10))
param_grid = {
	'criterion': ['gini','entropy'],
	'min_samples_split': [2] + list(range(25,501,25)),
	'min_samples_leaf': [1] + list(range(10,101,10)),
	'max_features': ['sqrt'] + list(range(1,len(X.columns)+1)),
	'max_depth': [1] + list(range(5,61,1)),
	'max_leaf_nodes': units + list(range(20,701,10))
}
print("\nParametros:")
print(param_grid)

#Inicializa a busca aleatória
print("\nInicializando a busca aleatória\n")
random_grid_search = RandomizedSearchCV(estimator=DT(),param_distributions=param_grid,n_iter=n,scoring='f1_weighted', n_jobs=-1, error_score='raise')
random_grid_search.fit(X,y)

#Obtém as combinações testadas e a F-score para cada uma
print("Obtendo as combinações testadas\n")
criterion = []
min_samples_split = []
min_samples_leaf = []
max_features = []
max_depth = []
max_leaf_nodes = []

'''
for combination in random_grid_search.cv_results_['params']:
  if combination['criterion'] == 'gini':
    criterion.append(0)
  else:
    criterion.append(1)
  min_samples_split.append(combination['min_samples_split'])
  min_samples_leaf.append(combination['min_samples_leaf'])
  max_features.append(combination['max_features'])
  max_depth.append(combination['max_depth'])
  max_leaf_nodes.append(combination['max_leaf_nodes'])
'''

for combination in random_grid_search.cv_results_['params']:
	if combination['criterion'] == 'gini':
		criterion.append(0)
	else:
		criterion.append(1)
	min_samples_split.append(combination['min_samples_split'])
	min_samples_leaf.append(combination['min_samples_leaf'])
	if (combination['max_features'] == 'sqrt'):
		max_features.append(0)
	else:
		max_features.append(combination['max_features'])
	max_depth.append(combination['max_depth'])
	max_leaf_nodes.append(combination['max_leaf_nodes'])

f1_scores = random_grid_search.cv_results_['mean_test_score']

print("Calculando a correlação\n")
#Calcula os coeficientes de correlação de Pearson entre os hiperparâmetros e os fscores obtidos
pr_criterion,_                  = pearsonr(criterion,f1_scores)
pr_min_samples_split,_          = pearsonr(min_samples_split,f1_scores)
pr_min_samples_leaf,_           = pearsonr(min_samples_leaf,f1_scores)
pr_max_features,_               = pearsonr(max_features,f1_scores)
pr_max_depth,_                  = pearsonr(max_depth,f1_scores)
pr_max_leaf_nodes,_             = pearsonr(max_leaf_nodes,f1_scores)

#Escreve em um arquivo as combinações testadas
print("Escrevendo as combinações testadas\n")
fid = open(model_name+"_random.csv","a")
fid.write("criterion,min_samples_split,min_samples_leaf,max_features,max_depth,max_leaf_nodes,f1\n")
for c,mss,msl,mf,md,mln,f1 in zip(criterion,min_samples_split,min_samples_leaf,max_features,max_depth,max_leaf_nodes,f1_scores):
	fid.write(str(c)+","+str(mss)+","+str(msl)+","+str(mf)+","+str(md)+","+str(mln)+","+str(f1)+"\n")
fid.close()

#Escreve em um arquivo a correlação entre os hiperparâmetros testados e a F-Score
print("Escrevendo a correlação entre os hiperparâmetros testados e a F-Score\n")
fid = open(model_name+"_correlacao.csv","a")
fid.write("criterion;min_samples_split;min_samples_leaf;max_features;max_depth;max_leaf_nodes\n")
fid.write(str(round(pr_criterion,4)).replace(".",",")+";"+str(round(pr_min_samples_split,4)).replace(".",",")+";"+str(round(pr_min_samples_leaf,4)).replace(".",",")+";"+str(round(pr_max_features,4)).replace(".",",")+";"+str(round(pr_max_depth,4)).replace(".",",")+";"+str(round(pr_max_leaf_nodes,4)).replace(".",",")+"\n")
fid.close()

#Escreve a melhor combinação encontrada pelo RandomSearch
print("Escrevendo a melhor combinação encontrada pelo RandomSearch\n")
fid = open(model_name+"_best_random.csv","w")
fid.write("criterion,min_samples_split,min_samples_leaf,max_features,max_depth,max_leaf_nodes,f1\n")
fid.write(str(random_grid_search.best_params_['criterion'])+","+str(random_grid_search.best_params_['min_samples_split'])+","+\
	str(random_grid_search.best_params_['min_samples_leaf'])+","+str(random_grid_search.best_params_['max_features'])+","+\
		str(random_grid_search.best_params_['max_depth'])+","+str(random_grid_search.best_params_['max_leaf_nodes'])+","+\
			str(random_grid_search.best_score_))
fid.close()

'''
	PARA EXECUTAR ESTE PROGRAMA, UTILIZE O SEGUINTE COMANDO:
		python3 executar_random.py
	EXEMPLO:
		python3 executar_random-RAMIRO.py
'''
import os
root = "/home/rgsviana/DATASETS_TZS/"
datasets_train = ["dataset_tzs_16x16_train.csv", "dataset_tzs_16x32_train.csv", "dataset_tzs_16x64_train.csv", "dataset_tzs_32x16_train.csv", "dataset_tzs_32x32_train.csv", "dataset_tzs_32x64_train.csv", "dataset_tzs_64x16_train.csv", "dataset_tzs_64x32_train.csv", "dataset_tzs_64x64_train.csv", "dataset_tzs_64x128_train.csv", "dataset_tzs_128x64_train.csv", "dataset_tzs_128x128_train.csv"]
datasets_test = ["dataset_tzs_16x16_test.csv", "dataset_tzs_16x32_test.csv", "dataset_tzs_16x64_test.csv", "dataset_tzs_32x16_test.csv", "dataset_tzs_32x32_test.csv", "dataset_tzs_32x64_test.csv", "dataset_tzs_64x16_test.csv", "dataset_tzs_64x32_test.csv", "dataset_tzs_64x64_test.csv", "dataset_tzs_64x128_test.csv", "dataset_tzs_128x64_test.csv", "dataset_tzs_128x128_test.csv"]
models = ["dataset_tzs_16x16", "dataset_tzs_16x32", "dataset_tzs_16x64", "dataset_tzs_32x16", "dataset_tzs_32x32", "dataset_tzs_32x64", "dataset_tzs_64x16", "dataset_tzs_64x32", "dataset_tzs_64x64", "dataset_tzs_64x128", "dataset_tzs_128x64", "dataset_tzs_128x128"]

for dataset_train, dataset_test, model in zip(datasets_train, datasets_test, models):
    os.system("python3 random_search-RAMIRO.py "+root+dataset_train+" "+root+dataset_test+" 1000 "+model)


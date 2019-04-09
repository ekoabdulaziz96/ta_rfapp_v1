from django.core.files.images import ImageFile
from django.shortcuts import render
from django.http import HttpResponse

from random_forest.models_dataset import Dataset as DatasetModel
from random_forest.models_setLabel import SetLabel as SetLabelModel
from random_forest.models_setFitur import SetFitur as SetFiturModel
from random_forest.models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel
from random_forest.models_randomForest import RandomForest as RandomForestModel
from random_forest import views
from django.http import JsonResponse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import seaborn as sns
sns.set()


def index(request):
    # judul = "<h1> halo bro</h1>"
    # sub_judul = "<h4> halo bro</h4>"

    context = {
        "title": "Blog",
        'heading': "welcome to my article",
        "judul": "Halaman Index Home",
        "kontributor": "MaseCoId_home",
        "banner": "img/banner_home.png"

    }

    # return render(request, 'templates.html', context)
    return render(request, 'index.html', context)


def coba(request):
    a = [
        [1, 2],
        [1, 2],
        [1, 2],
    ]
    df = pd.DataFrame(data=a, columns=['a', 'b'])
    # print(df)
    # df.to_csv('media/coba.csv', index=False)

    # input_file = 'media/dataset/csv/coba_new3.csv'
    # df = pd.read_csv(input_file, sep=',')
    # print(df)

    # dt = DatasetModel.objects.get(pk=4)
    # print(dt)
    # dt.file_dataset = 'coba.csv'
    # dt.save()
    # df = pd.read_csv(dt.file_dataset, sep=',')
    # print(df)

    from pylab import savefig
    i=5
    plt.figure(num=i,figsize=(20, 6), dpi=80)
    # plt.title("profil microRNA ke-{} ({})".format(i, profil[i]))
    # plt.grid(color='b', linestyle='-', linewidth=0.2)
    plot = sns.boxplot(data=df, orient="h", palette="Set2")
    plot = sns.swarmplot(data=df, orient="h")
    figure = plot.get_figure() 
    figure.savefig("media/output2.png")

    return HttpResponse('sukses')


def randomForest(request,rf_id, dataset_id, setlabel_id, setfitur_id, hyperparameter_id):
    return HttpResponse('sukses')

def get_x_y (df,get_label,get_fitur):
    df = df.fillna(value=0)

    X = df.drop([get_label.kolom_label], axis=1)
    for m in X.columns:
        if df[m].dtype == 'object':
            df = df.drop(m, axis=1)

    # context['out3'] = df.to_html(
    #     classes='table')

    if get_fitur.all_fitur == 1:
        X = df.drop([get_label.kolom_label], axis=1)
        # --- reduksi fitur dg semua nilai null/nol utk kelas tertentu
        if get_fitur.reduksi_null_fitur == 1:
            for m in X.columns:
                val = df.groupby(get_label.kolom_label)[m].sum()
                if val[0] == 0 or val[1] == 0:
                    df = df.drop(m, axis=1)

        
        # context['out4'] = df.to_html(
        #     classes='table')
        # --- reduksi fitur dg semua nilai null/nol utk kelas tertentu
        X = df.drop([get_label.kolom_label], axis=1)
        if int(get_fitur.reduksi_nilai_kurang_dari) > 0:
            for m in X.columns:
                if df[m].max() < int(get_fitur.reduksi_nilai_kurang_dari):
                    df = df.drop(m, axis=1)

        # context['out5'] = df.to_html(
        #     classes='table')
    else :
        # set fitur dan label
        fitur = get_fitur.fitur
        fitur = fitur.replace('[', '')
        fitur = fitur.replace(']', '')
        fitur = fitur.replace(' ', '')
        fitur = fitur.replace("'", '')
        fitur = fitur.split(',')
        if fitur[0] == '':
            fitur.remove('')
        
    y = df[get_label.kolom_label]
    X = df.drop([get_label.kolom_label], axis=1)
    if get_fitur.all_fitur == 0 :
        X=X[fitur]

    return X,y


def randomForest2(request,rf_id):
    get_rf = RandomForestModel.objects.get(
        pk=rf_id)
    get_dataset = get_rf.dataset
    get_label = get_rf.setlabel
    get_fitur = get_rf.setfitur
    get_hyperparameter = get_rf.hyperparameter

    file_result = 'media/randomForest/result/coba.csv'
    file_fitur_importance = 'media/randomForest/fiturImportance/coba.csv'
    file_model = 'media/randomForest/model/coba.pkl'

    file_result = file_result.replace('coba',str(get_rf.id))
    file_fitur_importance = file_fitur_importance.replace('coba',str(get_rf.id))
    file_model = file_model.replace('coba',str(get_rf.id))

    context = {}
    context['get_dataset'] = get_dataset
    context['get_label'] = get_label
    context['get_fitur'] = get_fitur
    context['get_hyperparameter'] = get_hyperparameter

    # To Dataframe
    df = views.dataframe(get_dataset.file_dataset, get_dataset.separator)
    context['out1'] = df.to_html(
        classes='table')

    # Preprocessing


    X,y = get_x_y(df,get_label,get_fitur)

    context['out6'] = X.to_html(
        classes='table')
    context['out7'] = list(y)
    # pemodelan
    maks_fitur = get_hyperparameter.max_fitur
    try:
        if isinstance(int(maks_fitur), int) == True:
            maks_fitur = int(maks_fitur)
    except:
        try:
            if isinstance(float(maks_fitur), float) == True:
                maks_fitur = float(maks_fitur)
        except Exception as e:
            pass


    clf = RandomForestClassifier(criterion='gini', max_depth=int(get_hyperparameter.max_kedalaman),
                                 max_features=maks_fitur, n_estimators=int(get_hyperparameter.n_tree), random_state=1)
    clf = clf.fit(X, y)
    pred = clf.predict(X)

    acc = accuracy_score(y, pred)
    cm_model = confusion_matrix(y, pred)

    # --K-FOld Cross Validation
    kfolds_cv = views.kfold_cross_validation(clf,X,y,n_fold=int(get_rf.k_cv),n_seed=1)
    df_result = pd.DataFrame(data= kfolds_cv, columns=['Accuracy','Sensitivity','Specifity','Time'])
    df_result.to_csv(file_result, index=False)
    
    # -- fitur importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fitur_importance = []
    for f in range(X.shape[1]):
        if  importances[indices[f]] > 0 :
            fitur_importance.append(
                [X.columns[indices[f]], importances[indices[f]]])

    df_FI = pd.DataFrame(data=fitur_importance, columns=['fitur', 'value'])
    df_FI.to_csv(file_fitur_importance, index=False)
    #model
    RFFile = open(file_model, 'wb')
    pickle.dump(clf, RFFile)
    RFFile.close()


    context['out8'] = clf
    context['out9'] = list(pred)
    context['out10'] = acc
    context['out11'] = cm_model
    context['out12'] = df_result.to_html(
        classes='table')
    context['out13'] = df_FI.to_html(
        classes='table')

    return render(request, 'coba.html', context)

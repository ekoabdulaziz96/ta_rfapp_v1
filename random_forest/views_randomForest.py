import random
import io
from pylab import savefig

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, render_to_response
from django.views.generic import ListView, DetailView, FormView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.messages.views import SuccessMessageMixin
from django.http import JsonResponse

from .models_dataset import Dataset as DatasetModel
from .models_setLabel import SetLabel as SetLabelModel
from .models_setFitur import SetFitur as SetFiturModel
from .models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel
from .models_randomForest import RandomForest as RandomForestModel
from .forms_randomForest import RandomForestForm
from . import views
from . import views_rf_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import json
# from IPython.display import Image  
from sklearn.externals.six import StringIO
from sklearn import tree  
import pydotplus
import pickle
import seaborn as sns
sns.set()


class RandomForestListView(ListView):
    model = RandomForestModel
    ordering = ['id']
    template_name = "random_forest/randomForest/index.html"
    context_object_name = 'randomforests'


    extra_context = {
        'page_judul': 'Tabel Run Random Forest',
        'page_deskripsi': 'mengelola Run Random Forest berdasarkan dataset default',
        'page_role': 'Run Random Forest',
    }

    def get_queryset(self):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            queryset = RandomForestModel.objects.filter(
                dataset_id=default_dataset.id)
        return queryset

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()
        count_default_hyperparameter = HyperparameterRFModel.objects.filter(
            default_hyperparameter=True).count()

        kwargs.update(self.extra_context)
        context = super(RandomForestListView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)

            context['set_hyperparameter'] = count_default_hyperparameter
        return context


class RandomForestCreateView(SuccessMessageMixin, CreateView):
    # model = RandomForestModel
    form_class = RandomForestForm
    template_name = "random_forest/randomForest/create.html"
    success_url = reverse_lazy('rf:random-forest-index')
    context_object_name = 'forms'

    extra_context = {
        'page_judul': 'Tambah Run Random Forest',
        'page_deskripsi': 'untuk menambah data Run Random Forest',
        'page_role': 'Run Random Forest',
    }

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()
        count_default_hyperparameter = HyperparameterRFModel.objects.filter(
            default_hyperparameter=True).count()

        kwargs.update(self.extra_context)
        context = super(RandomForestCreateView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1 and count_default_hyperparameter == 1:
            get_dataset = DatasetModel.objects.get(
                default_dataset=True)
            get_label = SetLabelModel.objects.filter(
                validate_label=True).filter(
                dataset_id=get_dataset.id).first()
            get_fitur = SetFiturModel.objects.filter(
                default_fitur=True).filter(
                dataset_id=get_dataset.id).first()
            get_hyperparameter = HyperparameterRFModel.objects.get(
                default_hyperparameter=True)
            
            df = views.dataframe(
                get_dataset.file_dataset, get_dataset.separator)

            X, y = views_rf_model.get_x_y(df, get_label, get_fitur)

            context['get_dataset'] = get_dataset
            context['get_label'] = get_label
            context['get_fitur'] = get_fitur
            context['get_hyperparameter'] = get_hyperparameter
            context['count_row_x'] = X.shape[0]

        return context

    def post(self, request, **kwargs):
        dataset_id = request.POST.get('dataset')
        setlabel_id = request.POST.get('setlabel')
        setfitur_id = request.POST.get('setfitur')
        hyperparameter_id = request.POST.get('hyperparameter')
        hyperparameter_id = request.POST.get('hyperparameter')
        k_cv = request.POST.get('k_cv')

        # print('dt_id', dataset_id)
        # print('sl_id', setlabel_id)
        # print('st_id', setfitur_id)
        # print('hp_id', hyperparameter_id)

        get_dataset = DatasetModel.objects.get(
            pk=dataset_id)
        get_label = SetLabelModel.objects.get(
            pk=setlabel_id)
        get_fitur = SetFiturModel.objects.get(
            pk=setfitur_id)
        get_hyperparameter = HyperparameterRFModel.objects.get(
            pk=hyperparameter_id)

        get_label.use_rf = True
        get_label.save()
        get_fitur.use_rf = True
        get_fitur.save()
        get_hyperparameter.use_rf = True
        get_hyperparameter.save()


        rf = RandomForestModel()
        rf.dataset_id = dataset_id
        rf.setlabel_id = setlabel_id
        rf.setfitur_id = setfitur_id
        rf.hyperparameter_id = hyperparameter_id
        rf.k_cv = k_cv
        rf.save()

        views_rf_model.randomForest(rf.id)

        file_result = 'randomForest/result/coba.csv'
        file_fitur_importance = 'randomForest/fiturImportance/coba.csv'
        file_model = 'randomForest/model/coba.pkl'

        file_result = file_result.replace('coba',str(rf.id))
        file_fitur_importance = file_fitur_importance.replace('coba',str(rf.id))
        file_model = file_model.replace('coba',str(rf.id))

        rf.rf_result = file_result
        rf.rf_fitur_importance = file_fitur_importance
        rf.rf_model = file_model
        rf.save()

        # success_url = self.get_success_url()
        return HttpResponseRedirect(reverse_lazy('rf:random-forest-index'))

        # return super(RandomForestCreateView,
                    #  self).post(request, **kwargs)

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil ditambahakan'


class RandomForestDeleteView(DeleteView):
    model = RandomForestModel
    # template_name = "random_forest/randomForest/create.html"
    success_url = reverse_lazy('rf:random-forest-index')

    def delete(self, request, *args, **kwargs):
        self.object = self.get_object()
        success_url = self.get_success_url()

        count_label = RandomForestModel.objects.filter(
            setlabel_id=self.object.setlabel_id).count()
        if count_label == 1:
            get_label = SetLabelModel.objects.get(
                pk=self.object.setlabel_id)
            get_label.use_rf = False
            get_label.save()
            # print('label')

        count_fitur = RandomForestModel.objects.filter(
            setfitur_id=self.object.setfitur_id).count()
        if count_fitur == 1:
            get_fitur = SetFiturModel.objects.get(
                pk=self.object.setfitur_id)
            get_fitur.use_rf = False
            get_fitur.save()
            # print('fitur')

        count_hyperparameter = RandomForestModel.objects.filter(
            hyperparameter_id=self.object.hyperparameter_id).count()
        if count_hyperparameter == 1:
            get_hyperparameter = HyperparameterRFModel.objects.get(
                pk=self.object.hyperparameter_id)
            get_hyperparameter.use_rf = False
            get_hyperparameter.save()
            # print('hyperparameter')

        self.object.delete()
        return HttpResponseRedirect(success_url)


class RandomForestDetailView(DetailView):
    model = RandomForestModel
    template_name = "random_forest/randomForest/detail.html"
    context_object_name = 'randomforest'

    extra_context = {
        'page_judul': 'Detail Run Random Forest',
        'page_deskripsi': 'untuk melihat detai data Run Random Forest',
        'page_role': 'Run Random Forest'
    }

    def get_context_data(self, *args, **kwargs):
        rf = RandomForestModel.objects.get(
            pk=self.kwargs.get('pk'))
        dataset = rf.dataset
        df = views.dataframe(dataset.file_dataset, dataset.separator)

        kwargs.update(self.extra_context)
        context = super(RandomForestDetailView,
                        self).get_context_data(*args, **kwargs)

        context['dataset_shape'] = df.shape
        context['label_label'] = list(df[rf.setlabel.kolom_label].unique())
        context['label_frekuensi'] = list(df[rf.setlabel.kolom_label].value_counts())

        df = df.fillna(value=0)

        X = df.drop([rf.setlabel.kolom_label], axis=1)
        # --- reduksi fitur dg semua nilai null/nol utk kelas tertentu
        if rf.setfitur.reduksi_null_fitur == 1:
            for m in X.columns:
                val = df.groupby(rf.setlabel.kolom_label)[m].sum()
                if val[0] == 0 or val[1] == 0:
                    df = df.drop(m, axis=1)

        X = df.drop([rf.setlabel.kolom_label], axis=1)
        # --- reduksi fitur dg semua nilai null/nol utk kelas tertentu
        if int(rf.setfitur.reduksi_nilai_kurang_dari) > 0:
            for m in X.columns:
                if df[m].max() < int(rf.setfitur.reduksi_nilai_kurang_dari):
                    df = df.drop(m, axis=1)

        context['dataset_shape_clean'] = df.shape

        df_result = views.dataframe(rf.rf_result,'koma')
        df_result.insert(loc=0, column='No', value=list(range(1,df_result.shape[0]+1)))
        df_result = df_result.set_index('No')
        df_result = df_result.append(df_result.describe()[1:2])
        context['df_result'] = df_result.to_html(
             classes='dataframe-style table')

        df_FI = views.dataframe(rf.rf_fitur_importance,'koma')
        df_FI.insert(loc=0, column='No', value=list(range(1,df_FI.shape[0]+1)))
        df_FI = df_FI.set_index('No')
        del df_FI.index.name
        context['df_FI'] = df_FI.to_html(
             classes='dataframe-style table')
        return context

class RandomForestDetailTreeView(DetailView):
    model = RandomForestModel
    template_name = "random_forest/randomForest/detail_tree.html"
    context_object_name = 'randomforest'

    extra_context = {
        'page_judul': 'Detail Run Random Forest',
        'page_deskripsi': 'untuk melihat detai data Run Random Forest',
        'page_role': 'Run Random Forest'
    }

    def get_context_data(self, *args, **kwargs):
        get_rf = RandomForestModel.objects.get(
            pk=self.kwargs.get('pk'))

        kwargs.update(self.extra_context)
        context = super(RandomForestDetailTreeView,
                        self).get_context_data(*args, **kwargs)

        context['dataset_shape'] = df.shape
        context['label_label'] = list(df[rf.setlabel.kolom_label].unique())
        context['label_frekuensi'] = list(df[rf.setlabel.kolom_label].value_counts())

        return context


def get_result(request, pk):
    rf = RandomForestModel.objects.get(pk=pk)
    df = views.dataframe(rf.rf_result,'koma')
    df.insert(loc=0, column='No', value=list(range(1,df.shape[0]+1)))
    df = df.set_index('No')
    df = df.append(df.describe()[1:2])
    data = df.to_html(
        classes='table table-striped text-center')
    return JsonResponse(data, safe=False)

def get_fitur_importance(request, pk):
    rf = RandomForestModel.objects.get(pk=pk)
    df = views.dataframe(rf.rf_fitur_importance,'koma')
    df.insert(loc=0, column='No', value=list(range(1,df.shape[0]+1)))
    df = df.set_index('No')
    del df.index.name
    data = df.to_html(
        classes='table table-striped text-center')
    return JsonResponse(data, safe=False)

def get_pohon(request, pk,no_pohon):
    get_rf = RandomForestModel.objects.get(
        pk=pk)
    context = {
        'page_judul': 'Detail Run Random Forest',
        'page_deskripsi': 'untuk melihat detai data Run Random Forest',
        'page_role': 'Run Random Forest',
        'no_pohon' : no_pohon,
        'randomforest': get_rf,
        'n_pohon' : list(range(int(get_rf.hyperparameter.n_tree)))
    }

    get_dataset = get_rf.dataset
    get_label = get_rf.setlabel
    get_fitur = get_rf.setfitur
    df = views.dataframe(get_dataset.file_dataset, get_dataset.separator)

    X,y = views_rf_model.get_x_y(df,get_label,get_fitur)
    print(str(get_rf.rf_model))
    # from pathlib import Path
    file_model = "media/"+str(get_rf.rf_model)
    RFFile = open(file_model,"rb")
    clf_rf = pickle.load(RFFile)

    dot_data = StringIO()  
    tree.export_graphviz(clf_rf.estimators_[no_pohon], out_file=dot_data,  
                                feature_names=X.columns)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    # Image(graph.create_png()) 
    graph.write_pdf("random_forest/static/random_forest/tree/tree.pdf")
    RFFile.close()

    return render(request, 'random_forest/randomForest/detail_tree.html', context)

import random
import io
from pylab import savefig
from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render, redirect, render_to_response
from django.views.generic import ListView, DetailView, FormView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.messages.views import SuccessMessageMixin
from django.core.cache import cache
from django.http import JsonResponse

from .models_dataset import Dataset as DatasetModel
from .models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel
from .forms_dataset import DatasetForm
from . import views

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import json

import seaborn as sns
sns.set()


class DatasetListView(ListView):
    model = DatasetModel
    ordering = ['id']
    template_name = "random_forest/dataset/index.html"
    context_object_name = 'datasets'

    extra_context = {
        'page_judul': 'Tabel Dataset',
        'page_deskripsi': 'untuk mengelola Dataset micro-RNA',
        'page_role': 'Dataset',
    }

    def get_context_data(self, *args, **kwargs):
        kwargs.update(self.extra_context)
        return super().get_context_data(*args, **kwargs)


class DatasetCreateView(SuccessMessageMixin, CreateView):
    # model = DatasetModel
    form_class = DatasetForm
    template_name = "random_forest/dataset/create.html"
    success_url = reverse_lazy('rf:dataset-index')
    context_object_name = 'forms'

    extra_context = {
        'page_judul': 'Tambah Dataset',
        'page_deskripsi': 'untuk menambah data Dataset',
        'page_role': 'Dataset',
    }

    def get_context_data(self, *args, **kwargs):
        kwargs.update(self.extra_context)
        return super().get_context_data(*args, **kwargs)

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil ditambahakan'


class DatasetUpdateView(SuccessMessageMixin, UpdateView):
    model = DatasetModel
    form_class = DatasetForm
    template_name = "random_forest/dataset/create.html"
    context_object_name = 'forms'
    success_url = reverse_lazy('rf:dataset-index')

    extra_context = {
        'page_judul': 'Edit Dataset',
        'page_deskripsi': 'untuk memperbarui data Dataset',
        'page_role': 'Dataset',
    }
    
    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        separator = request.POST.get('separator')
        dataset = self.object
        df = dataframe(dataset.file_dataset, separator)
        
        if (len(df.columns) == 1):
            self.object.default_dataset = False
            self.object.save()
        return super().post(request, *args, **kwargs)

    def get_context_data(self, *args, **kwargs):
        kwargs.update(self.extra_context)
        return super().get_context_data(*args, **kwargs)

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil diperbarui'


class DatasetDeleteView(DeleteView):
    # cache.clear()
    model = DatasetModel
    # template_name = "random_forest/dataset/create.html"
    success_url = reverse_lazy('rf:dataset-index')


def dataframe(file_dataset, separator):
    pd.set_option('colheader_justify', 'center')
    df = pd.DataFrame()
    if separator == 'koma':
        df = pd.read_csv(file_dataset, sep=',')
    elif separator == 'titik_koma':
        df = pd.read_csv(file_dataset, sep=';')
    elif separator == 'tab':
        df = pd.read_csv(file_dataset, sep='\n')
    elif separator == 'enter':
        df = pd.read_csv(file_dataset, sep='\t')

    return df


class DatasetDetailView(DetailView):
    model = DatasetModel
    template_name = "random_forest/dataset/detail.html"
    context_object_name = 'dataset'

    extra_context = {
        'page_judul': 'Detail Dataset',
        'page_deskripsi': 'untuk melihat detai data Dataset',
        'page_role': 'Dataset'
    }

    def get_context_data(self, *args, **kwargs):
        dataset = DatasetModel.objects.get(pk=self.kwargs.get('pk'))
        df = dataframe(dataset.file_dataset, dataset.separator)

        null = []
        for i in list(df.columns[:]):
            if df[i].isnull().sum() > 0 :
                null.append([i,df[i].isnull().sum()])
        df_null = pd.DataFrame(data=null,columns=['fitur','count_null'])

        kwargs.update(self.extra_context)
        context = super(DatasetDetailView, self).get_context_data(
            *args, **kwargs)

        context['df_null'] = df_null.to_html(
            classes='dataframe-style table')
        context['df_null_count'] = df_null.shape[0]
        context['df_head'] = df.head(20).to_html(
            classes='dataframe-style table')
        # context['df_tail'] = df.tail(20).to_html(
        #     classes='dataframe-style table')
        context['df_describe'] = df.describe().to_html(
            classes='dataframe-style-2 table')
        context['df_info'] = df.info()
        context['df_shape'] = df.shape
        # df = pd.DataFrame()

        return context

class DatasetDetailSwarmBoxView(DetailView):
    model = DatasetModel
    template_name = "random_forest/dataset/detail.html"
    context_object_name = 'dataset'

    extra_context = {
        'page_judul': 'Detail Dataset',
        'page_deskripsi': 'untuk melihat detai data Dataset',
        'page_role': 'Dataset'
    }

    def get_context_data(self, *args, **kwargs):
        dataset = DatasetModel.objects.get(pk=self.kwargs.get('pk'))
        df = dataframe(dataset.file_dataset, dataset.separator)

        kwargs.update(self.extra_context)
        context = super(DatasetDetailView, self).get_context_data(
            *args, **kwargs)

        context['df_shape'] = df.shape

        return context

def set_default(request, pk):
    if request.method == 'POST':
        dataset = DatasetModel.objects.get(pk=pk)
        df = views.dataframe(dataset.file_dataset, dataset.separator)
        if (len(df.columns) == 1):
            return JsonResponse('warning', safe=False)
        else:
            all_dataset = DatasetModel.objects.all()
            all_dataset.update(default_dataset=False)

            all_hyperparameter = HyperparameterRFModel.objects.all()
            all_hyperparameter.update(default_hyperparameter=False)

            dataset.default_dataset = True
            dataset.save()

            return JsonResponse('success', safe=False)

def boxplot(request, pk):
    dataset = DatasetModel.objects.get(pk=pk)
    df = dataframe(dataset.file_dataset, dataset.separator)
    df_new = df.copy()
    for i in list(df_new.columns[:]):
        if df[i].dtypes == 'O' :
            df_new = df_new.drop(i, axis=1)

    if df.shape[1] >= 5:
        random = np.random.choice(df.columns, 5, replace=False)
    else:
        random = np.random.choice(df.columns, df.shape[1], replace=False)
    df1 = df_new[random].transpose()

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    bplot=axes.boxplot(df1,labels=random ,vert=False)

    axes.set_title('Box-Plot', fontsize=14)

    html_fig = mpld3.fig_to_html(fig)

    return HttpResponse(html_fig)

def get_boxplot(request, pk,list_fitur):
    dataset = DatasetModel.objects.get(pk=pk)
    df = dataframe(dataset.file_dataset, dataset.separator)
    if list_fitur == 'default':
        df_new = df.copy()
        for i in list(df_new.columns[:]):
            if df[i].dtypes == 'O' :
                df_new = df_new.drop(i, axis=1)

        if df.shape[1] >= 5:
            random = np.random.choice(df.columns, 5, replace=False)
        else:
            random = np.random.choice(df.columns, df.shape[1], replace=False)
        df1 = df_new[random].transpose()
    else:
        fitur = list_fitur
        fitur = fitur.replace('[', '')
        fitur = fitur.replace(']', '')
        fitur = fitur.replace(' ', '')
        fitur = fitur.replace("'", '')
        fitur = fitur.split(",")
        # print(fitur)
        # print(len(fitur))
        df1 = df[fitur]

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    bplot=axes.boxplot(df1,labels=random ,vert=False)

    axes.set_title('Box-Plot', fontsize=14)

    html_fig = mpld3.fig_to_html(fig)

    return HttpResponse(html_fig)

def get_swarm_box(request, pk,list_fitur):
    get_dataset = DatasetModel.objects.get(
        pk=pk)

    df = views.dataframe(get_dataset.file_dataset, get_dataset.separator)
    df_new = df.copy()
    for i in list(df_new.columns[:]):
        if df[i].dtypes == 'object' :
            df_new = df_new.drop(i, axis=1)

    context = {
        'page_judul': 'Detail Dataset',
        'page_deskripsi': 'untuk melihat detai data Dataset',
        'page_role': 'Dataset',
        # 'no_pohon' : no_pohon,
        # 'randomforest': get_rf,
        # 'n_pohon' : list(range(int(get_rf.hyperparameter.n_tree)))
    }



    # graph.write_pdf("random_forest/static/random_forest/tree/tree.pdf")
    context['dataset'] = get_dataset
    context['fitur'] = df_new.columns
    context['url_plot'] = "/ML/dataset/detail/"+str(pk)+"/boxplot/"+str(list_fitur)
    return render(request, 'random_forest/dataset/detail_swarm_box.html', context)

import random
import io
from pylab import savefig
from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render, redirect, render_to_response
from django.views.generic import ListView, DetailView, FormView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.messages.views import SuccessMessageMixin
from django.http import JsonResponse
from django.core.serializers.json import DjangoJSONEncoder

from .models_dataset import Dataset as DatasetModel
from .models_setLabel import SetLabel as SetLabelModel
from .models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel
from .forms_setLabel import SetLabelForm
from . import views

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import json

import seaborn as sns
sns.set()


class SetLabelListView(ListView):
    model = SetLabelModel
    ordering = ['id']
    template_name = "random_forest/setLabel/index.html"
    context_object_name = 'setlabels'

    extra_context = {
        'page_judul': 'Tabel Set Label',
        'page_deskripsi': 'mengelola Default Dataset dan Label Dataset',
        'page_role': 'Set Label',
    }

    def get_queryset(self):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()
        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            queryset = SetLabelModel.objects.filter(
                dataset_id=default_dataset.id)
        return queryset

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()
        # print(count_default_dataset)

        kwargs.update(self.extra_context)
        context = super(SetLabelListView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)

            count_setlabel = SetLabelModel.objects.filter(
                dataset_id=default_dataset.id).count()
            context['count_setlabel'] = count_setlabel

        context['count_default_dataset'] = count_default_dataset
        return context


class SetLabelCreateView(SuccessMessageMixin, CreateView):
    # model = SetLabelModel
    form_class = SetLabelForm
    template_name = "random_forest/setLabel/create.html"
    success_url = reverse_lazy('rf:set-label-index')
    context_object_name = 'forms'

    extra_context = {
        'page_judul': 'Tambah Set Label',
        'page_deskripsi': 'untuk menambah data Set Label',
        'page_role': 'Set Label',
    }

    def get_context_data(self, *args, **kwargs):
        dataset = DatasetModel.objects.get(
            default_dataset=True)
        df = views.dataframe(dataset.file_dataset, dataset.separator)

        kwargs.update(self.extra_context)
        context = super(SetLabelCreateView,
                        self).get_context_data(*args, **kwargs)

        context['dataset'] = dataset
        context['all_fitur'] = df.columns
        return context

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil ditambahakan'


class SetLabelUpdateView(SuccessMessageMixin, UpdateView):
    model = SetLabelModel
    form_class = SetLabelForm
    template_name = "random_forest/setLabel/create.html"
    context_object_name = 'forms'
    success_url = reverse_lazy('rf:set-label-index')

    extra_context = {
        'page_judul': 'Edit Set Label',
        'page_deskripsi': 'untuk memperbarui data Set Label',
        'page_role': 'Set Label',
    }

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        kolom_label = request.POST.get('kolom_label')
        dataset = self.object.dataset
        df = views.dataframe(dataset.file_dataset, dataset.separator)
        if (len(df[kolom_label].unique()) != 2):
            self.object.validate_label = False
            self.object.save()

            self.object.dataset.set_label = False
            self.object.dataset.save()
        return super().post(request, *args, **kwargs)

    def get_context_data(self, *args, **kwargs):
        setlabel = SetLabelModel.objects.get(
            pk=self.kwargs.get('pk'))
        dataset = DatasetModel.objects.get(
            default_dataset=True)
        df = views.dataframe(dataset.file_dataset, dataset.separator)
        labels = df[setlabel.kolom_label].unique

        kwargs.update(self.extra_context)
        context = super(SetLabelUpdateView,
                        self).get_context_data(*args, **kwargs)
        context['setlabel'] = setlabel
        context['labelkanker'] = setlabel.nilai_label_kanker
        context['labels'] = labels
        context['dataset'] = dataset
        context['all_fitur'] = df.columns
        return context

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil diperbarui'


class SetLabelDeleteView(DeleteView):
    model = SetLabelModel
    # template_name = "random_forest/setLabel/create.html"
    success_url = reverse_lazy('rf:set-label-index')


class SetLabelDetailView(DetailView):
    model = SetLabelModel
    template_name = "random_forest/setLabel/detail.html"
    context_object_name = 'setlabel'

    extra_context = {
        'page_judul': 'Detail Set Label',
        'page_deskripsi': 'untuk melihat detai data Set Label',
        'page_role': 'Set Label'
    }

    def get_context_data(self, *args, **kwargs):
        setLabel = SetLabelModel.objects.get(
            pk=self.kwargs.get('pk'))
        # print(setLabel)
        dataset = setLabel.dataset
        df = views.dataframe(dataset.file_dataset, dataset.separator)

        kwargs.update(self.extra_context)
        context = super(SetLabelDetailView, self).get_context_data(
            *args, **kwargs)

        context['X_shape'] = '"--pastikan mengisi nilai kolom label  &  nilai label kelas kanker dengan benar--"'
        context['count_label'] = '"--pastikan mengisi nilai kolom label &  nilai label kelas kanker dengan benar--"'

        # if (setLabel.kolom_label in df.columns) and (setLabel.nilai_label_kanker in df[setLabel.kolom_label].unique()):
        if (len(df[setLabel.kolom_label].unique()) == 2):
            df_new = df.drop(setLabel.kolom_label, axis=1)
            count_label = df[setLabel.kolom_label].value_counts()

            context['X_shape'] = df_new.shape[1]
            context['count_label'] = count_label
            context['count'] = list(count_label)
            context['label'] = list(df[setLabel.kolom_label].unique())

        return context


def boxplot(request, pk):
    dataset = SetLabelModel.objects.get(pk=pk)
    df = views.dataframe(dataset.file_dataset, dataset.separator)

    for i in list(df.columns[:]):
        if df[i].dtypes == 'O':
            df_new = df.drop(i, axis=1)

    if df.shape[1] >= 5:
        random = np.random.choice(df.columns, 5, replace=False)
    else:
        random = np.random.choice(df.columns, df.shape[1], replace=False)
    df1 = df_new[random].transpose()

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    bplot = axes.boxplot(df1, labels=random, vert=False)

    axes.set_title('Box-Plot', fontsize=14)

    html_fig = mpld3.fig_to_html(fig)

    return HttpResponse(html_fig)


def set_default(request, pk):
    if request.method == 'POST':
        setLabel = SetLabelModel.objects.get(pk=pk)
        dataset = setLabel.dataset
        df = views.dataframe(dataset.file_dataset, dataset.separator)
        if (len(df[setLabel.kolom_label].unique()) != 2):
            return JsonResponse('warning', safe=False)
        else:
            # all_setLabel = SetLabelModel.objects.all()
            # all_setLabel.update(validate_label=False)

            setLabel.validate_label = True
            setLabel.save()

            dataset.set_label = True
            dataset.save()

            return JsonResponse('success', safe=False)


def get_label(request, id_dataset):
    dataset = DatasetModel.objects.get(pk=id_dataset)
    df = views.dataframe(dataset.file_dataset, dataset.separator)

    dataset_serialized = json.dumps(list(df.columns), cls=DjangoJSONEncoder)

    return JsonResponse(dataset_serialized, safe=False)


def get_label_kanker(request, id_dataset, kolom_label):
    dataset = DatasetModel.objects.get(pk=id_dataset)
    df = views.dataframe(dataset.file_dataset, dataset.separator)
    val_label = df[kolom_label].unique()

    if val_label.dtype != 'object':
        val_label = val_label.astype('object')

    # print(val_label.dtype)
    val_label_serialized = json.dumps(list(val_label), cls=DjangoJSONEncoder)

    return JsonResponse(val_label_serialized, safe=False)

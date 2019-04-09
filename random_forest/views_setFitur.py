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

from .models_dataset import Dataset as DatasetModel
from .models_setLabel import SetLabel as SetLabelModel
from .models_setFitur import SetFitur as SetFiturModel
from .forms_setFitur import SetFiturForm
from . import views

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import json

import seaborn as sns
sns.set()


class SetFiturListView(ListView):
    model = SetFiturModel
    ordering = ['id']
    template_name = "random_forest/setFitur/index.html"
    context_object_name = 'setfiturs'

    extra_context = {
        'page_judul': 'Tabel Set Fitur',
        'page_deskripsi': 'mengelola Set Fitur berdasarkan dataset default',
        'page_role': 'Set Fitur',
    }

    def get_queryset(self):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            queryset = SetFiturModel.objects.filter(
                dataset_id=default_dataset.id)
        return queryset

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        kwargs.update(self.extra_context)
        context = super(SetFiturListView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            set_label = SetLabelModel.objects.filter(
                validate_label=True).filter(dataset_id=default_dataset.id).count()

            context['set_label'] = set_label
        return context


class SetFiturCreateView(SuccessMessageMixin, CreateView):
    # model = SetFiturModel
    form_class = SetFiturForm
    template_name = "random_forest/setFitur/create.html"
    success_url = reverse_lazy('rf:set-fitur-index')
    context_object_name = 'forms'

    extra_context = {
        'page_judul': 'Tambah Set Fitur',
        'page_deskripsi': 'untuk menambah data Set Fitur',
        'page_role': 'Set Fitur',
    }

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        kwargs.update(self.extra_context)
        context = super(SetFiturCreateView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            get_dataset = SetLabelModel.objects.get(
                dataset_id=default_dataset.id)
            dataset = get_dataset.dataset
            df = views.dataframe(dataset.file_dataset, dataset.separator)
            X = df.drop(get_dataset.kolom_label, axis=1)

            context['get_dataset'] = get_dataset
            context['fiturs'] = X.columns
        return context

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil ditambahakan'


class SetFiturUpdateView(SuccessMessageMixin, UpdateView):
    model = SetFiturModel
    form_class = SetFiturForm
    template_name = "random_forest/setFitur/create.html"
    context_object_name = 'forms'
    success_url = reverse_lazy('rf:set-fitur-index')

    extra_context = {
        'page_judul': 'Edit Set Fitur',
        'page_deskripsi': 'untuk memperbarui data Set Fitur',
        'page_role': 'Set Fitur',
    }

    def post(self, request, **kwargs):
        self.object = self.get_object()

        fitur = self.object.fitur
        fitur = fitur.replace('[', '')
        fitur = fitur.replace(']', '')
        fitur = fitur.replace(' ', '')
        fitur = fitur.replace("'", '')

        mutable = request.POST._mutable
        request.POST._mutable = True

        if request.POST.get('fitur') == '':
            request.POST['fitur'] = fitur
        request.POST._mutable = mutable
        return super(SetFiturUpdateView,
                     self).post(request, **kwargs)

    def get_context_data(self, *args, **kwargs):
        setfitur = SetFiturModel.objects.get(
            pk=self.kwargs.get('pk'))
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        kwargs.update(self.extra_context)
        context = super(SetFiturUpdateView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            get_dataset = SetLabelModel.objects.get(
                dataset_id=default_dataset.id)
            dataset = get_dataset.dataset
            df = views.dataframe(dataset.file_dataset, dataset.separator)
            X = df.drop(get_dataset.kolom_label, axis=1)

            context['get_dataset'] = get_dataset
            context['fiturs'] = X.columns

        context['setfitur'] = setfitur
        return context

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil diperbarui'


class SetFiturDeleteView(DeleteView):
    model = SetFiturModel
    # template_name = "random_forest/setFitur/create.html"
    success_url = reverse_lazy('rf:set-fitur-index')


class SetFiturDetailView(DetailView):
    model = SetFiturModel
    template_name = "random_forest/setFitur/detail.html"
    context_object_name = 'setfitur'

    extra_context = {
        'page_judul': 'Detail Set Fitur',
        'page_deskripsi': 'untuk melihat detai data Set Fitur',
        'page_role': 'Set Fitur'
    }

    def get_context_data(self, *args, **kwargs):

        kwargs.update(self.extra_context)
        context = super(SetFiturDetailView, self).get_context_data(
            *args, **kwargs)

        return context


def set_default(request, pk):
    if request.method == 'POST':
        setfitur = SetFiturModel.objects.get(pk=pk)

        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            all_setfitur = SetFiturModel.objects.filter(dataset_id = default_dataset.id)
            all_setfitur.update(default_fitur=False)

        setfitur.default_fitur = True
        setfitur.save()

        return JsonResponse('success', safe=False)

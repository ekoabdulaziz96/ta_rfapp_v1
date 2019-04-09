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

from django.http import JsonResponse
from .models_dataset import Dataset as DatasetModel
from .models_setLabel import SetLabel as SetLabelModel
from .models_setFitur import SetFitur as SetFiturModel
from .models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel
from .forms_hyperparameterRF import HyperparameterRFForm

from . import views
from . import views_rf_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import json

import seaborn as sns
sns.set()


class HyperparameterRFListView(ListView):
    model = HyperparameterRFModel
    ordering = ['id']
    template_name = "random_forest/hyperparameterRF/index.html"
    context_object_name = 'hyperparameterRFs'

    extra_context = {
        'page_judul': 'Tabel Hyperparameter RF',
        'page_deskripsi': 'mengelola Table Set Hyperparameter Random Forest',
        'page_role': 'Set Hyperparameter RF',
    }

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        kwargs.update(self.extra_context)
        context = super(HyperparameterRFListView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            count_fitur = SetFiturModel.objects.filter(
                default_fitur=True).filter(dataset_id=default_dataset.id).count()
            if count_fitur == 1:
                get_label = SetLabelModel.objects.filter(
                    validate_label=True).filter(dataset_id=default_dataset.id).first()
                get_fitur = SetFiturModel.objects.filter(
                    default_fitur=True).filter(dataset_id=default_dataset.id).first()
                df = views.dataframe(
                    default_dataset.file_dataset, default_dataset.separator)

                X, y = views_rf_model.get_x_y(df, get_label, get_fitur)

                context['count_x'] = X.shape[1]
            context['count_fitur'] = count_fitur
        return context


class HyperparameterRFCreateView(SuccessMessageMixin, CreateView):
    # model = HyperparameterRFModel
    form_class = HyperparameterRFForm
    template_name = "random_forest/hyperparameterRF/create.html"
    success_url = reverse_lazy('rf:hyperparameter-RF-index')
    context_object_name = 'forms'

    extra_context = {
        'page_judul': 'Tambah Set Hyperparameter RF',
        'page_deskripsi': 'untuk menambah data Set Hyperparameter RF',
        'page_role': 'Set Hyperparameter RF',
    }

    def post(self, request, **kwargs):
        # self.object = self.get_object()

        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            count_fitur = SetFiturModel.objects.filter(
                default_fitur=True).filter(dataset_id=default_dataset.id).count()
            if count_fitur == 1:
                get_label = SetLabelModel.objects.filter(
                    validate_label=True).filter(dataset_id=default_dataset.id).first()
                get_fitur = SetFiturModel.objects.filter(
                    default_fitur=True).filter(dataset_id=default_dataset.id).first()
                df = views.dataframe(
                    default_dataset.file_dataset, default_dataset.separator)

                X, y = views_rf_model.get_x_y(df, get_label, get_fitur)

        mutable = request.POST._mutable
        request.POST._mutable = True

        try:
            if isinstance(int(request.POST.get('max_fitur')), int) == True:
                if int(request.POST.get('max_fitur')) > X.shape[1]:
                    request.POST['max_fitur'] = X.shape[1]
                    print('cek')
        except Exception as e:
            pass

        request.POST._mutable = mutable
        return super(HyperparameterRFCreateView,
                     self).post(request, **kwargs)

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        kwargs.update(self.extra_context)
        context = super(HyperparameterRFCreateView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            count_fitur = SetFiturModel.objects.filter(
                default_fitur=True).filter(dataset_id=default_dataset.id).count()
            if count_fitur == 1:
                get_label = SetLabelModel.objects.filter(
                    validate_label=True).filter(dataset_id=default_dataset.id).first()
                get_fitur = SetFiturModel.objects.filter(
                    default_fitur=True).filter(dataset_id=default_dataset.id).first()
                df = views.dataframe(
                    default_dataset.file_dataset, default_dataset.separator)

                X, y = views_rf_model.get_x_y(df, get_label, get_fitur)

                context['count_x'] = X.shape[1]
        return context

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil ditambahakan'


class HyperparameterRFUpdateView(SuccessMessageMixin, UpdateView):
    model = HyperparameterRFModel
    form_class = HyperparameterRFForm
    template_name = "random_forest/hyperparameterRF/create.html"
    context_object_name = 'forms'
    success_url = reverse_lazy('rf:hyperparameter-RF-index')

    extra_context = {
        'page_judul': 'Edit Set Hyperparameter RF',
        'page_deskripsi': 'untuk memperbarui data Set Hyperparameter RF',
        'page_role': 'Set Hyperparameter RF',
    }

    def post(self, request, **kwargs):
        # self.object = self.get_object()

        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            count_fitur = SetFiturModel.objects.filter(
                default_fitur=True).filter(dataset_id=default_dataset.id).count()
            if count_fitur == 1:
                get_label = SetLabelModel.objects.filter(
                    validate_label=True).filter(dataset_id=default_dataset.id).first()
                get_fitur = SetFiturModel.objects.filter(
                    default_fitur=True).filter(dataset_id=default_dataset.id).first()
                df = views.dataframe(
                    default_dataset.file_dataset, default_dataset.separator)

                X, y = views_rf_model.get_x_y(df, get_label, get_fitur)

        mutable = request.POST._mutable
        request.POST._mutable = True

        try:
            if isinstance(int(request.POST.get('max_fitur')), int) == True:
                if int(request.POST.get('max_fitur')) > X.shape[1]:
                    request.POST['max_fitur'] = X.shape[1]
                    print('cek')
        except Exception as e:
            pass

        request.POST._mutable = mutable
        return super(HyperparameterRFUpdateView,
                     self).post(request, **kwargs)

    def get_context_data(self, *args, **kwargs):
        count_default_dataset = DatasetModel.objects.filter(
            default_dataset=True).count()

        kwargs.update(self.extra_context)
        context = super(HyperparameterRFUpdateView,
                        self).get_context_data(*args, **kwargs)

        if count_default_dataset == 1:
            default_dataset = DatasetModel.objects.get(
                default_dataset=True)
            count_fitur = SetFiturModel.objects.filter(
                default_fitur=True).filter(dataset_id=default_dataset.id).count()
            if count_fitur == 1:
                get_label = SetLabelModel.objects.filter(
                    validate_label=True).filter(dataset_id=default_dataset.id).first()
                get_fitur = SetFiturModel.objects.filter(
                    default_fitur=True).filter(dataset_id=default_dataset.id).first()
                df = views.dataframe(
                    default_dataset.file_dataset, default_dataset.separator)

                X, y = views_rf_model.get_x_y(df, get_label, get_fitur)

                context['count_x'] = X.shape[1]
        return context

    def get_success_message(self, cleaned_data):
        return 'Data Dataset berhasil diperbarui'


class HyperparameterRFDeleteView(DeleteView):
    model = HyperparameterRFModel
    # template_name = "random_forest/hyperparameterRF/create.html"
    success_url = reverse_lazy('rf:hyperparameter-RF-index')


class HyperparameterRFDetailView(DetailView):
    model = HyperparameterRFModel
    template_name = "random_forest/hyperparameterRF/detail.html"
    context_object_name = 'hyperparameterRF'

    extra_context = {
        'page_judul': 'Detail Set Hyperparameter RF',
        'page_deskripsi': 'untuk melihat detai data Set Hyperparameter RF',
        'page_role': 'Set Hyperparameter RF'
    }

    def get_context_data(self, *args, **kwargs):

        kwargs.update(self.extra_context)
        context = super(HyperparameterRFDetailView, self).get_context_data(
            *args, **kwargs)

        return context


def set_default(request, pk):
    if request.method == 'POST':
        set_hyperparameter = HyperparameterRFModel.objects.get(pk=pk)

        all_set_hyperparameter = HyperparameterRFModel.objects.all()
        all_set_hyperparameter.update(default_hyperparameter=False)

        set_hyperparameter.default_hyperparameter = True
        set_hyperparameter.save()

        return JsonResponse('success', safe=False)

from django.urls import path, include
from django.views.generic.base import TemplateView, RedirectView
from django.views.generic import ListView, DetailView, FormView

from . import views_dataset, views_setLabel
from . import views_setFitur, views_hyperparameterRF
from . import views_randomForest

app_name = 'random_forest'

urlpatterns = [
    path('random-forest/pohon/<int:pk>/<int:no_pohon>',
         views_randomForest.get_pohon, name='random-forest-pohon'),
    path('random-forest/fitur_importance/<int:pk>',
         views_randomForest.get_fitur_importance, name='random-forest-fitur-importance'),
    path('random-forest/result/<int:pk>',
         views_randomForest.get_result, name='random-forest-result'),
    path('random-forest/detail-tree/<int:pk>',
         views_randomForest.RandomForestDetailTreeView.as_view(), name='random-forest-detail-tree'),
    path('random-forest/detail/<int:pk>',
         views_randomForest.RandomForestDetailView.as_view(), name='random-forest-detail'),
    path('random-forest/delete/<int:pk>',
         views_randomForest.RandomForestDeleteView.as_view(), name='random-forest-delete'),
    path('random-forest/create/', views_randomForest.RandomForestCreateView.as_view(),
         name='random-forest-create'),
    path('random-forest/', views_randomForest.RandomForestListView.as_view(),
         name='random-forest-index'),

    # -------------------set hyperparameter RF
    path('hyperparameter-RF/default/<int:pk>',
         views_hyperparameterRF.set_default, name='hyperparameter-RF-set-default'),
    path('hyperparameter-RF/detail/<int:pk>',
         views_hyperparameterRF.HyperparameterRFDetailView.as_view(), name='hyperparameter-RF-detail'),
    path('hyperparameter-RF/update/<int:pk>',
         views_hyperparameterRF.HyperparameterRFUpdateView.as_view(), name='hyperparameter-RF-update'),
    path('hyperparameter-RF/delete/<int:pk>',
         views_hyperparameterRF.HyperparameterRFDeleteView.as_view(), name='hyperparameter-RF-delete'),
    path('hyperparameter-RF/create/', views_hyperparameterRF.HyperparameterRFCreateView.as_view(),
         name='hyperparameter-RF-create'),
    path('hyperparameter-RF/', views_hyperparameterRF.HyperparameterRFListView.as_view(),
         name='hyperparameter-RF-index'),

    # -------------------set fitur
    path('set-fitur/default/<int:pk>',
         views_setFitur.set_default, name='set-fitur-set-default'),
    path('set-fitur/detail/<int:pk>',
         views_setFitur.SetFiturDetailView.as_view(), name='set-fitur-detail'),
    path('set-fitur/update/<int:pk>',
         views_setFitur.SetFiturUpdateView.as_view(), name='set-fitur-update'),
    path('set-fitur/delete/<int:pk>',
         views_setFitur.SetFiturDeleteView.as_view(), name='set-fitur-delete'),
    path('set-fitur/create/', views_setFitur.SetFiturCreateView.as_view(),
         name='set-fitur-create'),
    path('set-fitur/', views_setFitur.SetFiturListView.as_view(),
         name='set-fitur-index'),

    # -------------------set label
    path('set-label/create/label-kanker/<int:id_dataset>/<str:kolom_label>',
         views_setLabel.get_label_kanker, name='set-label-get-label-kanker'),
    path('set-label/create/label/<int:id_dataset>',
         views_setLabel.get_label, name='set-label-get-label'),
    path('set-label/default/<int:pk>',
         views_setLabel.set_default, name='set-label-set-default'),
    path('set-label/detail/<int:pk>',
         views_setLabel.SetLabelDetailView.as_view(), name='set-label-detail'),
    path('set-label/update/<int:pk>',
         views_setLabel.SetLabelUpdateView.as_view(), name='set-label-update'),
    path('set-label/delete/<int:pk>',
         views_setLabel.SetLabelDeleteView.as_view(), name='set-label-delete'),
    path('set-label/create/', views_setLabel.SetLabelCreateView.as_view(),
         name='set-label-create'),
    path('set-label/', views_setLabel.SetLabelListView.as_view(),
         name='set-label-index'),

    # -------------------dataset
    path('dataset/default/<int:pk>',
         views_dataset.set_default, name='dataset-set-default'),
    path('dataset/detail/<int:pk>/boxplot/<str:list_fitur>',
         views_dataset.get_boxplot, name='dataset-detail-get-boxplot'),
    path('dataset/detail/<int:pk>/boxplot',
         views_dataset.boxplot, name='dataset-detail-boxplot'),
    path('dataset/detail-swarm-box/<int:pk>/<str:list_fitur>',
         views_dataset.get_swarm_box, name='dataset-detail-swarm-box'),
    path('dataset/detail/<int:pk>',
         views_dataset.DatasetDetailView.as_view(), name='dataset-detail'),
    path('dataset/update/<int:pk>',
         views_dataset.DatasetUpdateView.as_view(), name='dataset-update'),
    path('dataset/delete/<int:pk>',
         views_dataset.DatasetDeleteView.as_view(), name='dataset-delete'),
    path('dataset/create/', views_dataset.DatasetCreateView.as_view(),
         name='dataset-create'),
    path('dataset/', views_dataset.DatasetListView.as_view(), name='dataset-index'),


    #     path('detail/<int:id_buku>', views.detail, name='detail'),
    #     path('update/<int:id_buku>', views.update, name='update'),
    #     path('delete/<int:id_buku>', views.delete, name='delete'),
    #     path('create-file/', views.create_file, name='create-file'),
    #     path('create/', views.create, name='create'),
    #     path('', views.index, name='index'),

]

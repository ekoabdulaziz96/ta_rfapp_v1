# membuat form
from django import forms

from .models_dataset import Dataset as DatasetModel


class DatasetForm(forms.ModelForm):
    error_css_class = 'error'

    class Meta:
        model = DatasetModel
        fields = (
            'nama',
            'deskripsi',
            'file_dataset',
            'separator',
        )

        labels = {
            'file_dataset': 'File Dataset (*.csv)',
            'separator': 'Separator File (*.csv)',
        }

        widgets = {
            'nama': forms.TextInput(
                attrs={
                    'class': 'form-control',
                    'placeholder': 'nama dataset',
                }
            ),
            'deskripsi': forms.Textarea(
                attrs={
                    'class': 'form-control',
                    'placeholder': 'deskripsi/informasi dataset',
                }
            ),
            'file_dataset': forms.ClearableFileInput(
                attrs={
                    'class': 'form-control',
                }
            ),
            'separator': forms.Select(
                attrs={
                    'class': 'form-control',

                }
            ),

        }

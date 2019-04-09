# membuat form
from django import forms

from .models_hyperparameterRF import HyperparameterRF as HyperparameterRFModel


class HyperparameterRFForm(forms.ModelForm):
    error_css_class = 'error'

    class Meta:
        model = HyperparameterRFModel
        fields = (
            'n_tree',
            'max_fitur',
            'max_kedalaman',
        )

        labels = {
            'n_tree': 'Jumlah Pohon dalam Random Forest',
            'max_fitur': 'Maksimal Fitur (saat split pohon)',
            'max_kedalaman': 'Maksimal Kedalaman Pohon dalam Random Forest',
        }

        widgets = {
            'n_tree': forms.NumberInput(
                attrs={
                    'class': 'form-control',
                    'min': 1
                }
            ),
            'max_fitur': forms.TextInput(
                attrs={
                    'class': 'form-control',
                }
            ),
            'max_kedalaman': forms.NumberInput(
                attrs={
                    'class': 'form-control',
                    'placeholder': '50',
                    'min': 1

                }
            ),
        }

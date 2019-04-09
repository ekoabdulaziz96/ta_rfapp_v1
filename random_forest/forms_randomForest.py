# membuat form
from django import forms

from .models_randomForest import RandomForest as RandomForestModel


class RandomForestForm(forms.ModelForm):
    error_css_class = 'error'

    class Meta:
        model = RandomForestModel
        fields = (
            'dataset',
            'setlabel',
            'setfitur',
            'hyperparameter',
            'k_cv',
        )

        labels = {
            'setlabel': ' Label(y)',
            'setfitur': 'Fitur(X)',
            'hyperparameter': 'Hyperparameter Random Forest',
            'k_cv': 'Nilai " k " dalam K-Fold Cross Validation',
        }

        widgets = {
            'dataset': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'setlabel': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'setfitur': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'hyperparameter': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'k_cv': forms.NumberInput(
                attrs={
                    'class': 'form-control',
                    'min' : 2
                }
            ),
        }

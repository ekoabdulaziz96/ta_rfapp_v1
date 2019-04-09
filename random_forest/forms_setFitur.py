# membuat form
from django import forms

from .models_setFitur import SetFitur as SetFiturModel


class SetFiturForm(forms.ModelForm):
    error_css_class = 'error'

    class Meta:
        model = SetFiturModel
        fields = (
            'dataset',
            'all_fitur',
            'fitur',
            'reduksi_null_fitur',
            'reduksi_nilai_kurang_dari',
        )

        labels = {
            'all_fitur': 'Gunakan Semua Fitur',
            'reduksi_null_fitur': 'Eliminasi Fitur yang semua nilainya adalah null/nol untuk kelas tertentu',
            'reduksi_nilai_kurang_dari': 'Eliminasi Fitur dengan maks nilai kurang dari',
        }

        widgets = {
            'dataset': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'all_fitur': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'fitur': forms.SelectMultiple(
                attrs={
                    'class': 'form-control',
                }
            ),
            'reduksi_null_fitur': forms.Select(
                attrs={
                    'class': 'form-control',
                }
            ),
            'reduksi_nilai_kurang_dari': forms.NumberInput(
                attrs={
                    'class': 'form-control',
                    'placeholder': '50',
                    'min': 0

                }
            ),
        }

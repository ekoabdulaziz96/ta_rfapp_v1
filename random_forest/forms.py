# membuat form
from django import forms

from .models import Buku as BukuModel


# class BukuForm(forms.ModelForm):
#     class Meta:
#         model = BukuModel
#         fields = (
#             'title',
#             'author',
#             'file_data',
#             'cover'
#         )

#         labels = {
#             'author': 'Penulis',
#             'file_data': 'File',
#             'cover': 'Sampul Buku',
#         }

#         widgets = {
#             'title': forms.TextInput(
#                 attrs={
#                     'class': 'form-control',
#                     'placeholder': 'judul buku',
#                 }
#             ),
#             'author': forms.TextInput(
#                 attrs={
#                     'class': 'form-control',
#                     'placeholder': 'nama penulis',
#                 }
#             ),
#             'file_data': forms.ClearableFileInput(
#                 attrs={
#                     'class': 'form-control',
#                 }
#             ),
#             'cover': forms.ClearableFileInput(
#                 attrs={
#                     'class': 'form-control',
#                 }
#             ),

#         }

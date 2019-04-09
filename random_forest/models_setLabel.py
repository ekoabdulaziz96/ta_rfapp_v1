from django.db import models
from django.utils.text import slugify

from .models_dataset import Dataset as DatasetModel

# def validate_file_extension(value):
#     import os
#     from django.core.exceptions import ValidationError
#     ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
#     valid_extensions = ['.csv']
#     if not ext.lower() in valid_extensions:
#         raise ValidationError(
#             u'mohon maaf, hanya mendukung file dengan ekstensi .csv')


class SetLabel(models.Model):
    dataset = models.OneToOneField(DatasetModel, on_delete=models.CASCADE)
    validate_label = models.BooleanField(default=False)
    use_rf = models.BooleanField(default=False)
    kolom_label = models.CharField(max_length=250)
    nilai_label_kanker = models.CharField(max_length=255)

    # separator = models.CharField(
    #     max_length=255,
    #     default="koma",
    #     choices=(
    #         ('koma', 'Koma'),
    #         ('titik_koma', 'Titik Koma'),
    #         ('tab', 'Tab'),
    #         ('enter', 'Enter'),
    #     )
    # )

    def __str__(self):
        return "[{}] -> {}".format(self.id, self.kolom_label)

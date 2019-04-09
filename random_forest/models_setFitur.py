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


class SetFitur(models.Model):
    # reduksi_null_fitur = models.BooleanField(default=False)
    dataset = models.ForeignKey(
        DatasetModel, on_delete=models.CASCADE)
    fitur = models.TextField(blank=True, null=True)
    default_fitur = models.BooleanField(default=False)
    use_rf = models.BooleanField(default=False)
    all_fitur = models.BooleanField(
        max_length=255,
        default=True,
        choices=(
            (True, 'Ya'),
            (False, 'Tidak'),
        )
    )
    reduksi_null_fitur = models.BooleanField(
        max_length=255,
        default=True,
        choices=(
            (True, 'Ya'),
            (False, 'Tidak'),
        )
    )
    reduksi_nilai_kurang_dari = models.CharField(max_length=255,default=0)

    def __str__(self):
        return "[{}] -> {}".format(self.id, self.fitur)

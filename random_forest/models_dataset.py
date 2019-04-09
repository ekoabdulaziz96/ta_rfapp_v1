from django.db import models
from django.utils.text import slugify


def validate_file_extension(value):
    import os
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.csv']
    if not ext.lower() in valid_extensions:
        raise ValidationError(
            u'FILE DATASET : mohon maaf, hanya mendukung file dengan ekstensi .csv')


class Dataset(models.Model):
    nama = models.CharField(max_length=255, unique=True)
    deskripsi = models.TextField()
    default_dataset = models.BooleanField(default=False)
    set_label = models.BooleanField(default=False)
    file_dataset = models.FileField(
        upload_to='dataset/csv/', validators=[validate_file_extension])
    # cover = models.ImageField(upload_to='buku/cover/', null=True, blank=True)
    separator = models.CharField(
        max_length=255,
        default="koma",
        choices=(
            ('koma', 'Koma'),
            ('titik_koma', 'Titik Koma'),
            ('tab', 'Tab'),
            ('enter', 'Enter'),
        )
    )

    def delete(self, *args, **kwargs):
        self.file_dataset.delete()
        super().delete(*args, **kwargs)

    def __str__(self):
        return "[{}] {}".format(self.id, self.nama)

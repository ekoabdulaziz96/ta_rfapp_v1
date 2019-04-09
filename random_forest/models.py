from django.db import models
from django.utils.text import slugify


# def validate_file_extension(value):
#     import os
#     from django.core.exceptions import ValidationError
#     ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
#     valid_extensions = ['.pdf']
#     if not ext.lower() in valid_extensions:
#         raise ValidationError(u'Unsupported file extension.only PDF files')


# class Buku(models.Model):
#     title = models.CharField(max_length=255)
#     author = models.CharField(max_length=255)
#     file_data = models.FileField(
#         upload_to='buku/pdf/', validators=[validate_file_extension])
#     cover = models.ImageField(upload_to='buku/cover/', null=True, blank=True)

#     def delete(self, *args, **kwargs):
#         self.file_data.delete()
#         self.cover.delete()
#         super().delete(*args, **kwargs)

#     def __str__(self):
#         return "[{}] {}".format(self.id, self.title)

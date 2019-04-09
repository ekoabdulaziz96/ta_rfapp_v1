from django.db import models
from django.utils.text import slugify
from django.core.exceptions import ValidationError


def validate_max_fitur(value):
    valid_string = ['auto', 'sqrt', 'log2']
    validasi = False

    try:
        if isinstance(int(value), int) == True:
            validasi = True
            # print(int(value))
            if int(value) < 1:
                validasi = 'err_int'
    except:
        try:
            if isinstance(float(value), float) == True:
                validasi = True
#                 print(float(value) < 0.1)
                if float(value) < 0.1:
                    validasi = 'err_float'
        except Exception as e:
            pass

    if validasi == 'err_int':
        raise ValidationError(
            u'MAKS FITUR : maaf, format integer (angka)  harus positif dan lebih besar atau sama dengan 1')
    elif validasi == 'err_float':
        raise ValidationError(
            u'MAKS FITUR : maaf, format float harus positif dan lebih besar atau sama dengan 0.1')
    elif value not in valid_string and validasi == False:
        raise ValidationError(
            u'MAKS FITUR : maaf, format string yang dibolehkan: auto,sqrt,log2 & silahkan gunakan titik(.) untuk format float')


class HyperparameterRF(models.Model):
    use_rf = models.BooleanField(default=False)
    default_hyperparameter = models.BooleanField(default=False)
    n_tree = models.CharField(max_length=255, default=10)
    max_fitur = models.CharField(
        max_length=255, default='sqrt', validators=[validate_max_fitur])
    max_kedalaman = models.CharField(max_length=255, default=10)

    def __str__(self):
        return "[{}] -> n_tree({}) ,max_fitur({}), max_kedalaman({})".format(self.id, self.n_tree, self.max_fitur, self.max_kedalaman)
